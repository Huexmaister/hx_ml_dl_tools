from typing import Any, Optional, Union
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap
import tensorflow as tf

from info_tools import InfoTools

warnings.filterwarnings("ignore")


class DlShapTools:
    """
    Clase para generar visualizaciones SHAP para modelos Keras / TensorFlow (TF 2.20).
    Diseñada para comportarse similar a tu MlShapTools pero adaptada a modelos deep learning.
    """

    # --------------------------------------------------------------------------------------------
    # -- 0: Constructor
    # --------------------------------------------------------------------------------------------
    def __init__(self,
                 x_test: pd.DataFrame,
                 model_name: str,
                 save_path: str,
                 model_object: Union[tf.keras.Model, Any],
                 sample: bool = True,
                 num_features_to_show: int = 100,
                 num_sample: int = 200,
                 background_sample: int = 100):
        """
        :param x_test: pd.DataFrame - Datos de prueba (features) para explicar (sin target).
        :param model_name: str - Nombre del modelo.
        :param save_path: str - Ruta base donde se guardarán los gráficos (se creará subcarpeta SHAP).
        :param model_object: tf.keras.Model (modelo ya cargado/entrenado).
        :param sample: bool - Si se usa muestreo para x_test (reduce coste computacional).
        :param num_features_to_show: int - Top N features a mostrar en gráficas.
        :param num_sample: int - Número máximo de filas a conservar en x_test si sample=True.
        :param background_sample: int - Número de instancias para background (explicador deep/grad).
        """
        # ---- 0.1: Herramientas de utilidad
        self.IT: InfoTools = InfoTools()

        # ---- 0.2: Guardado de parámetros y tipado
        self.x_test: pd.DataFrame = x_test.copy()
        self.model_name: str = model_name
        self.save_path: str = os.path.join(save_path, "SHAP")
        self.sample: bool = sample
        self.num_features_to_show: int = min(num_features_to_show, len(self.x_test.columns))
        self.num_sample: int = num_sample
        self.background_sample: int = background_sample
        self.model: Union[tf.keras.Model, Any] = model_object

        # ---- 0.3: Crear directorio
        os.makedirs(self.save_path, exist_ok=True)

        # ---- 0.4: Mensaje inicial
        self.IT.sub_intro_print(f"Realizando análisis SHAP (DL) y generando gráficos para {self.model_name}...")

        # ---- 0.5: Muestreo si procede
        if self.sample and len(self.x_test) > self.num_sample:
            try:
                self.x_test = self.x_test.sample(n=self.num_sample, random_state=42)
            except Exception as e:
                print(f"Error en el muestreo de x_test: {e}")

        # --------------------------------------------------------------------------------------------
        # -- 1: Preparar datos de background y predicciones (para clasificar/regresión)
        # --------------------------------------------------------------------------------------------
        # ---- 1.1: Background para explainer (submuestra de x_test, o si no hay suficientes filas, usar todas)
        if len(self.x_test) > 0:
            self._background: pd.DataFrame = self.x_test.sample(n=min(self.background_sample, len(self.x_test)), random_state=42)
        else:
            raise ValueError("x_test no puede estar vacío para generar explicaciones SHAP.")

        # ---- 1.2: Inferir tipo de problema y normalizar predict_proba function
        self.problem_type: str = "unknown"  # 'regression' | 'binary' | 'multiclass'
        self._prepare_model_prediction_interface()

        # --------------------------------------------------------------------------------------------
        # -- 2: Inicializar SHAP Explainer y calcular valores SHAP
        # --------------------------------------------------------------------------------------------
        self.explainer: Optional[Any] = None
        self.shap_values_raw: Optional[Any] = None  # resultado directo del explainer
        self.shap_values: Optional[np.ndarray] = None  # numpy procesado
        self.expected_value: Optional[Union[float, np.ndarray]] = None
        self.shap_df: Optional[pd.DataFrame] = None
        self.shap_sum: Optional[pd.Series] = None

        self._initialize_shap_explainer()

    # --------------------------------------------------------------------------------------------
    # -- 1.1: Preparar función de predicción (normalizar salidas del modelo)
    # --------------------------------------------------------------------------------------------
    def _prepare_model_prediction_interface(self) -> None:
        """
        Normaliza la interfaz de predicción del modelo Keras:
        - define self._model_predict(X) -> np.ndarray de probabilidades/valores
        - define self._choose_positive_class(shap_values) -> devuelve hx_shap_tools para clase positiva si procede
        También intenta identificar si el problema es binario/multiclass/regresión.
        """
        # ---- 1.1.1: función de predicción que siempre acepta numpy array o DataFrame
        def model_predict_numpy(x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
            # Aceptar DataFrame: pasar valores al modelo
            if isinstance(x, pd.DataFrame):
                arr = x.values
            else:
                arr = np.asarray(x)

            # Usar model.predict con verbose=0 para Keras
            preds = self.model.predict(arr, verbose=0)

            # Asegurar numpy array
            preds = np.asarray(preds)

            return preds

        self._model_predict = model_predict_numpy

        # ---- 1.1.2: inspeccionar una predicción pequeña para inferir tipo
        try:
            sample_pred = self._model_predict(self._background.iloc[:min(5, len(self._background))])
        except Exception as e:
            raise RuntimeError(f"Error ejecutando model.predict sobre background: {e}")

        # ---- 1.1.3: inferir tipo en base a la forma de la salida
        if sample_pred.ndim == 1:
            # Salida 1D -> regresión o probabilidad directa (caso raro)
            self.problem_type = "regression"
        elif sample_pred.ndim == 2:
            ncols = sample_pred.shape[1]
            if ncols == 1:
                # Forma (n,1) -> regresión o probabilidad binaria (positiva en [:,0])
                # Decidimos: si el rango 0..1 lo consideramos probabilidad; si no, regresión.
                mn, mx = float(np.min(sample_pred)), float(np.max(sample_pred))
                if 0.0 <= mn and mx <= 1.0:
                    self.problem_type = "binary"
                else:
                    self.problem_type = "regression"
            elif ncols == 2:
                # (n,2) -> típicamente clasificación binaria con 2 probabilidades (softmax)
                self.problem_type = "binary"
            else:
                # (n, k>2) -> multiclass
                self.problem_type = "multiclass"
        elif sample_pred.ndim == 3:
            # Casos raros (por ejemplo, salidas con más dimensiones) -> tratamos como regression multi-dim
            self.problem_type = "regression"
        else:
            self.problem_type = "regression"

        self.IT.info_print(f"Tipo de problema detectado: {self.problem_type}")

    # --------------------------------------------------------------------------------------------
    # -- 2: Inicializar SHAP Explainer
    # --------------------------------------------------------------------------------------------
    def _initialize_shap_explainer(self) -> None:
        """
        Inicializa el explainer de SHAP para modelos Keras.
        Se intenta usar la API unificada `hx_shap_tools.Explainer` con un masker apropiado.
        Si falla, intento `hx_shap_tools.DeepExplainer` como fallback.
        Finalmente se calculan los valores SHAP y se procesan para uso posterior.
        """
        try:
            # ---- 2.1: Preparar una función de predicción que devuelva la forma esperada por SHAP
            def f_model(x_array: np.ndarray) -> np.ndarray:
                # hx_shap_tools.Explainer puede pasar pandas DataFrame o numpy
                preds = self._model_predict(x_array)
                return preds

            # ---- 2.2: intentar crear un Explainer genérico (API unificada)
            try:
                masker = shap.maskers.Independent(self._background.values)
                self.explainer = shap.Explainer(f_model, masker, output_names=None)
            except Exception:
                # fallback: intentar DeepExplainer
                try:
                    self.IT.info_print("hx_shap_tools.Explainer con masker falló, intentando DeepExplainer fallback...")
                    # DeepExplainer pide background como array
                    self.explainer = shap.DeepExplainer(self.model, self._background.values)
                except Exception as e_deep:
                    raise RuntimeError(f"No se pudo inicializar explainer: {e_deep}")

            # ---- 2.3: calcular hx_shap_tools values (esto puede tardar)
            self.IT.info_print("Calculando valores SHAP (puede tardar según tamaño de muestra)...")
            shap_result = self.explainer(self.x_test.values)

            # ---- 2.4: almacenar raw y procesar
            self.shap_values_raw = shap_result

            # expected_value
            try:
                # hx_shap_tools.Explainer devuelve object con base_values o expected_value
                if hasattr(shap_result, "base_values"):
                    self.expected_value = shap_result.base_values
                elif hasattr(self.explainer, "expected_value"):
                    self.expected_value = self.explainer.expected_value
                else:
                    self.expected_value = None
            except Exception:
                self.expected_value = None

            # ---- 2.5: normalizar shap_result a numpy (compatibilizar con código anterior)
            self._process_shap_values()

            # ---- 2.6: construir DataFrame y suma de importancias
            # Para features mostrar: top num_features_to_show
            self.shap_df = pd.DataFrame(
                data=self._shap_values_2d_for_df(self.shap_values),
                columns=self.x_test.columns,
                index=self.x_test.index
            )

            # mean absolute SHAP
            self.shap_sum = np.abs(self.shap_df).mean().sort_values(ascending=False)
            self.shap_sum = self.shap_sum[: self.num_features_to_show]

            self.IT.info_print("Explainer inicializado y valores SHAP calculados OK.")

        except Exception as e:
            print(f"Error inicializando SHAP explainer: {e}")
            raise

    # --------------------------------------------------------------------------------------------
    # -- 2.1: Procesar hx_shap_tools raw según tipo de salida
    # --------------------------------------------------------------------------------------------
    def _process_shap_values(self) -> None:
        """
        Convierte self.shap_values_raw (hx_shap_tools.Explanation u otros) a numpy arrays y establece
        self.shap_values con forma controlada:
          - regression or binary with single output: (n_samples, n_features)
          - binary with 2-prob outputs: (n_samples, n_features) -> valores para la clase positiva
          - multiclass: (n_samples, n_classes, n_features)
        """
        raw = self.shap_values_raw

        # ---- 2.1.1: si es un objeto Explanation (API unificada)
        if hasattr(raw, "values"):
            vals = raw.values  # puede ser (n, features) o (n, classes, features)
        else:
            vals = raw  # dejar lo que venga (lista o array)

        vals = np.asarray(vals)

        # ---- 2.1.2: Casos:
        if vals.ndim == 2:
            # (n_samples, n_features)
            self.shap_values = vals
        elif vals.ndim == 3:
            # (n_samples, n_classes, n_features)
            # Si multiclass -> guardamos tal cual
            self.shap_values = vals
        elif isinstance(vals, list):
            # Lista de arrays por clase (posible en algunos explainers)
            try:
                stacked = np.stack(vals, axis=1)  # (n_samples, n_classes, n_features)
                self.shap_values = stacked
            except Exception:
                # fallback: intentar convertir a (n_samples, n_features) tomando el primer elemento
                self.shap_values = np.asarray(vals[0])
        else:
            # fallback: tratar como 2D
            self.shap_values = vals.reshape(vals.shape[0], -1)

        # ---- 2.1.3: almacenar shap_values_class_positive: matriz (n_samples, n_features)
        if self.problem_type == "binary":
            # Si self.shap_values es (n, n_classes, n_features) -> tomar clase 1 si existe
            if self.shap_values.ndim == 3:
                # preferimos la clase 1 como "positiva" si existe
                n_classes = self.shap_values.shape[1]
                class_positive_idx = 1 if n_classes > 1 else 0
                self.shap_values_class_positive: np.ndarray = self.shap_values[:, class_positive_idx, :]
            else:
                # (n, features) ya
                self.shap_values_class_positive = self.shap_values
        elif self.problem_type == "multiclass":
            # mantener la estructura (n_samples, n_classes, n_features)
            self.shap_values_class_positive = self.shap_values
        else:
            # regression
            self.shap_values_class_positive = self.shap_values

        # Exponer como atributo
        self.shap_values_processed: np.ndarray = self.shap_values_class_positive

    # --------------------------------------------------------------------------------------------
    # -- Util: convertir shap_values a 2D para DataFrame si procede
    # --------------------------------------------------------------------------------------------
    @staticmethod
    def _shap_values_2d_for_df(shap_vals: np.ndarray) -> np.ndarray:
        """
        Convierte shap_vals a (n_samples, n_features) tomando la clase positiva si shap_vals tiene
        dimensión de clase.
        """
        if shap_vals is None:
            raise ValueError("shap_vals es None")

        if shap_vals.ndim == 2:
            return shap_vals
        elif shap_vals.ndim == 3:
            # tomar la clase positiva si existe (index 1 si hay >=2 clases)
            n_classes = shap_vals.shape[1]
            idx = 1 if n_classes > 1 else 0
            return shap_vals[:, idx, :]
        else:
            # reshape fallback
            n_samples = shap_vals.shape[0]
            return shap_vals.reshape(n_samples, -1)

    # --------------------------------------------------------------------------------------------
    # -- get_feature_importance_summary
    # --------------------------------------------------------------------------------------------
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """
        Retorna un DataFrame con resumen de importancia: feature, mean_abs_shap, mean_shap, std_shap
        """
        summary_df = pd.DataFrame({
            "feature": self.shap_sum.index,
            "mean_abs_shap": self.shap_sum.values,
            "mean_shap": self.shap_df[self.shap_sum.index].mean().values,
            "std_shap": self.shap_df[self.shap_sum.index].std().values
        })
        return summary_df

    # --------------------------------------------------------------------------------------------
    # -- plot_summary
    # --------------------------------------------------------------------------------------------
    def plot_summary(self, plot_type: str = "violin") -> None:
        """
        Genera el gráfico summary de SHAP (violin o dot) y lo guarda como PNG.
        :param plot_type: "violin" o "dot"
        """
        try:
            plt.figure(figsize=(10, 8))
            # hx_shap_tools.summary_plot acepta Explanation o arrays
            shap.summary_plot(
                self.shap_values_processed if self.problem_type != "multiclass" else self.shap_values_processed[:, 0, :],
                self.x_test,
                show=False,
                plot_type=plot_type,
                max_display=self.num_features_to_show
            )
            plt.tight_layout()
            filename = os.path.join(self.save_path, f"{self.model_name}_shap_summary_{plot_type}.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            self.IT.info_print(f"Gráfico summary guardado: {filename}")
        except Exception as e:
            print(f"Error generando summary plot: {e}")

    # --------------------------------------------------------------------------------------------
    # -- save_barplot
    # --------------------------------------------------------------------------------------------
    def save_barplot(self, show_figure: bool = False) -> None:
        """
        Genera y guarda un barplot (HTML) con la importancia (mean |SHAP|).
        """
        try:
            fig_bar = go.Figure([
                go.Bar(
                    x=self.shap_sum.values,
                    y=self.shap_sum.index,
                    orientation="h",
                    marker_color="rgba(55, 128, 191, 0.7)",
                    marker_line=dict(color="rgba(55, 128, 191, 1.0)", width=1)
                )
            ])

            fig_bar.update_layout(
                title=f"Feature Importance - {self.model_name} (Top {len(self.shap_sum)})",
                xaxis_title="Mean |SHAP value|",
                yaxis_title="Features",
                height=max(400, len(self.shap_sum) * 20),
                margin=dict(l=150, r=50, t=80, b=50)
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"))

            filename = os.path.join(self.save_path, f"{self.model_name}_shap_barplot_importance.html")
            fig_bar.write_html(filename)

            if show_figure:
                fig_bar.show()

            self.IT.info_print(f"Gráfico de barras guardado: {filename}")
        except Exception as e:
            print(f"Error generando barplot: {e}")

    # --------------------------------------------------------------------------------------------
    # -- save_boxplot
    # --------------------------------------------------------------------------------------------
    def save_boxplot(self, show_figure: bool = False) -> None:
        """
        Genera y guarda un boxplot interactivo (HTML) con la distribución de valores SHAP por feature.
        """
        try:
            fig_box = go.Figure()

            # generar una paleta reproducible con matplotlib
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.shap_sum.index)))

            for i, feature in enumerate(self.shap_sum.index):
                color_rgb = (int(colors[i][0] * 255), int(colors[i][1] * 255), int(colors[i][2] * 255))
                fig_box.add_trace(
                    go.Box(
                        y=self.shap_df[feature],
                        name=feature,
                        marker_color=f"rgba({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}, 0.7)"
                    )
                )

            fig_box.update_layout(
                title=f"SHAP Values Distribution - {self.model_name}",
                xaxis_title="Features",
                yaxis_title="SHAP value",
                height=600,
                showlegend=False
            )

            if len(self.shap_sum.index) > 10:
                fig_box.update_layout(xaxis_tickangle=-45)

            filename = os.path.join(self.save_path, f"{self.model_name}_shap_boxplot.html")
            fig_box.write_html(filename)

            if show_figure:
                fig_box.show()

            self.IT.info_print(f"Boxplot guardado: {filename}")
        except Exception as e:
            print(f"Error generando boxplot: {e}")

    # --------------------------------------------------------------------------------------------
    # -- save_waterfall_plot
    # --------------------------------------------------------------------------------------------
    def save_waterfall_plot(self, instance_idx: int = 0) -> None:
        """
        Genera gráfico waterfall para una instancia específica y lo guarda como PNG.
        """
        try:
            # Preparar Explanation para la instancia
            if hasattr(self.shap_values_raw, "values") and hasattr(self.shap_values_raw, "data"):
                # si usamos Explanation de hx_shap_tools.Explainer
                # construir Explanation para una instancia
                if self.shap_values_processed.ndim == 2:
                    vals_inst = self.shap_values_processed[instance_idx]
                else:
                    # multiclass -> tomar clase positiva (idx 1 si existe)
                    if self.shap_values_processed.ndim == 3:
                        n_classes = self.shap_values_processed.shape[1]
                        idx = 1 if n_classes > 1 else 0
                        vals_inst = self.shap_values_processed[instance_idx, idx, :]
                    else:
                        vals_inst = self.shap_values_processed[instance_idx]

                base_val = self.expected_value if not isinstance(self.expected_value, (list, np.ndarray)) else (self.expected_value[1] if getattr(self.expected_value, "__len__", None) and len(self.expected_value) > 1 else self.expected_value[0])

                expl = shap.Explanation(values=vals_inst,
                                       base_values=base_val,
                                       data=self.x_test.iloc[instance_idx].values,
                                       feature_names=list(self.x_test.columns))
            else:
                # fallback: crear Explanation mínimo
                if self.shap_values_processed.ndim == 2:
                    vals_inst = self.shap_values_processed[instance_idx]
                else:
                    vals_inst = self.shap_values_processed[instance_idx, 0, :]
                base_val = self.expected_value if self.expected_value is not None else 0.0
                expl = shap.Explanation(values=vals_inst,
                                       base_values=base_val,
                                       data=self.x_test.iloc[instance_idx].values,
                                       feature_names=list(self.x_test.columns))

            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(expl, show=False)
            filename = os.path.join(self.save_path, f"{self.model_name}_shap_waterfall_instance_{instance_idx}.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            self.IT.info_print(f"Waterfall plot guardado: {filename}")

        except Exception as e:
            print(f"Error generando waterfall plot: {e}")

    # --------------------------------------------------------------------------------------------
    # -- save_force_plot
    # --------------------------------------------------------------------------------------------
    def save_force_plot(self, instance_idx: int = 0) -> None:
        """
        Genera force plot para una instancia y la guarda como HTML.
        """
        try:
            # expected value
            expected_value = self.expected_value
            if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

            # valores para la instancia
            if self.shap_values_processed.ndim == 2:
                vals_inst = self.shap_values_processed[instance_idx]
            else:
                vals_inst = self.shap_values_processed[instance_idx, 0, :]

            force_plot = shap.force_plot(
                expected_value,
                vals_inst,
                self.x_test.iloc[instance_idx],
                show=False
            )

            filename = os.path.join(self.save_path, f"{self.model_name}_shap_force_instance_{instance_idx}.html")
            shap.save_html(filename, force_plot)
            self.IT.info_print(f"Force plot guardado: {filename}")

        except Exception as e:
            print(f"Error generando force plot: {e}")

    # --------------------------------------------------------------------------------------------
    # -- run (ejecuta pipeline completo y guarda outputs)
    # --------------------------------------------------------------------------------------------
    def run(self, include_waterfall: bool = True, include_force: bool = True) -> pd.DataFrame:
        """
        Ejecuta todos los métodos de visualización y guarda un CSV con el resumen de importancias.
        Retorna summary_df.
        """
        # ---- 1: Gráficos principales
        self.plot_summary("violin")
        self.plot_summary("dot")
        self.save_barplot()
        self.save_boxplot()

        # ---- 2: Gráficos para instancias concretas (si hay datos)
        if include_waterfall and len(self.x_test) > 0:
            self.save_waterfall_plot(0)

        if include_force and len(self.x_test) > 0:
            self.save_force_plot(0)

        # ---- 3: Guardar resumen
        summary_df = self.get_feature_importance_summary()
        summary_filename = os.path.join(self.save_path, f"{self.model_name}_shap_feature_importance_summary.csv")
        summary_df.to_csv(summary_filename, index=False)
        self.IT.info_print(f"Dataframe con las importancias relativas guardado: {summary_filename}")

        return summary_df
