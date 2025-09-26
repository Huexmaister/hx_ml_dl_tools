from info_tools import InfoTools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap
import os
import warnings

warnings.filterwarnings('ignore')


class MlShapTools:
    def __init__(self,
                 x_test: pd.DataFrame,
                 model_name: str,
                 save_path: str,
                 model_object,
                 sample: bool = True,
                 num_features_to_show: int = 100,
                 num_sample: int = 500):
        """
        Clase para generar visualizaciones SHAP para modelos de clasificación de machine learning
        :param x_test: pd.DataFrame - Datos de prueba para explicar
        :param model_name: Nombre del modelo
        :param save_path: Ruta donde guardar los gráficos
        :param model_object: instancia deñ modelo entrenado (XGBoost, LightGBM, CatBoost, RandomForest, etc.)
        :param sample: Si se debe hacer muestreo de los datos
        :param num_features_to_show: Número de características a mostrar en los gráficos (las n primeras)
        :param num_sample: Número de muestras a usar si sample=True
        """

        self.IT: InfoTools = InfoTools()

        # Almaceno parámetros en propiedades
        self.x_test: pd.DataFrame = x_test.copy()
        self.model_name: str = self._normalize_model_name(model_name)
        self.save_path: str = f"{save_path}/SHAP"
        self.sample: bool = sample
        self.num_features_to_show: int = min(num_features_to_show, len(x_test.columns))
        self.num_sample: int = num_sample
        self.model = model_object

        # Crear directorio si no existe
        os.makedirs(self.save_path, exist_ok=True)

        # - Pinto la entrada
        self.IT.sub_intro_print(f"Realizando análisis SHAP y generando gráficos....")

        # Sampleo si es necesario
        if sample and len(self.x_test) > num_sample:
            try:
                self.x_test = self.x_test.sample(n=num_sample, random_state=42)
            except ValueError as e:
                print(f"Error en el muestreo: {e}")

        # Inicializar explainer y valores SHAP
        self._initialize_shap_explainer()

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """Normaliza el nombre del modelo."""
        model_name_lower = model_name.lower()
        if "xgb" in model_name_lower or "xgboost" in model_name_lower:
            return "XGBoost"
        elif "lightgbm" in model_name_lower or "lgb" in model_name_lower:
            return "LightGBM"
        elif "catboost" in model_name_lower or "cat" in model_name_lower:
            return "CatBoost"
        elif "randomforest" in model_name_lower or "rf" in model_name_lower:
            return "RandomForest"
        else:
            return model_name

    def _initialize_shap_explainer(self):
        """Inicializa el explainer de SHAP y calcula los valores SHAP."""
        try:
            # Crear explainer
            self.explainer = shap.TreeExplainer(self.model)

            # Calcular valores SHAP
            self.shap_values = self.explainer.shap_values(self.x_test)

            # Manejar diferentes tipos de salidas de modelos
            self._process_shap_values()

            # Crear DataFrame con valores SHAP
            self.shap_df = pd.DataFrame(
                self.shap_values_class_positive,
                columns=self.x_test.columns,
                index=self.x_test.index
            )

            # Calcular importancia promedio absoluta
            self.shap_sum = np.abs(self.shap_df).mean().sort_values(ascending=False)
            self.shap_sum = self.shap_sum[:self.num_features_to_show]

        except Exception as e:
            print(f"Error inicializando SHAP explainer: {e}")
            raise

    def _process_shap_values(self):
        """Procesa los valores SHAP según el tipo de modelo y problema."""
        if isinstance(self.shap_values, list):
            # Para modelos de clasificación multiclase o binaria con múltiples salidas
            if len(self.shap_values) == 2:
                # Clasificación binaria: tomar valores de la clase positiva (índice 1)
                self.shap_values_class_positive = self.shap_values[1]

            else:
                # Clasificación multiclase: tomar la primera clase o hacer promedio
                self.shap_values_class_positive = self.shap_values[0]

        else:
            # Para modelos con salida única (XGBoost binario, regresión, etc.)
            self.shap_values_class_positive = self.shap_values

    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Retorna un resumen de la importancia de las características."""
        summary_df = pd.DataFrame({
            'feature': self.shap_sum.index,
            'mean_abs_shap': self.shap_sum.values,
            'mean_shap': self.shap_df[self.shap_sum.index].mean(),
            'std_shap': self.shap_df[self.shap_sum.index].std()
        })
        return summary_df

    def plot_summary(self, plot_type: str = "violin"):
        """
        Genera el gráfico summary de SHAP.

        Parameters:
        -----------
        plot_type : str, default="violin"
            Tipo de gráfico: "violin" o "dot"
        """
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values_class_positive,
                self.x_test,
                show=False,
                plot_type=plot_type,
                max_display=self.num_features_to_show
            )

            plt.tight_layout()
            filename = f'{self.save_path}/{self.model_name}_shap_summary_{plot_type}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            self.IT.info_print(f"Gráfico summary guardado: {filename}")

        except Exception as e:
            print(f"Error generando summary plot: {e}")

    def save_barplot(self, show_figure: bool = False):
        """Genera y guarda gráfico de barras con importancia de características."""
        try:
            fig_bar = go.Figure([
                go.Bar(
                    x=self.shap_sum.values,
                    y=self.shap_sum.index,
                    orientation='h',
                    marker_color='rgba(55, 128, 191, 0.7)',
                    marker_line=dict(color='rgba(55, 128, 191, 1.0)', width=1)
                )
            ])

            fig_bar.update_layout(
                title=f'Feature Importance - {self.model_name} (Top {len(self.shap_sum)})',
                xaxis_title='Mean |SHAP value|',
                yaxis_title='Features',
                height=max(400, len(self.shap_sum) * 20),
                margin=dict(l=150, r=50, t=80, b=50)
            )

            # Invertir orden del eje Y para mostrar la característica más importante arriba
            fig_bar.update_layout(yaxis=dict(autorange="reversed"))

            filename = f'{self.save_path}/{self.model_name}_shap_barplot_importance.html'
            fig_bar.write_html(filename)

            if show_figure:
                fig_bar.show()

            self.IT.info_print(f"Gráfico de barras guardado: {filename}")

        except Exception as e:
            print(f"Error generando barplot: {e}")

    def save_boxplot(self, show_figure: bool = False):
        """Genera y guarda boxplot de valores SHAP por característica."""
        try:
            fig_box = go.Figure()

            colors = plt.cm.Set3(np.linspace(0, 1, len(self.shap_sum.index)))

            for i, feature in enumerate(self.shap_sum.index):
                fig_box.add_trace(
                    go.Box(
                        y=self.shap_df[feature],
                        name=feature,
                        marker_color=f'rgba({int(colors[i][0] * 255)}, {int(colors[i][1] * 255)}, {int(colors[i][2] * 255)}, 0.7)'
                    )
                )

            fig_box.update_layout(
                title=f'SHAP Values Distribution - {self.model_name}',
                xaxis_title='Features',
                yaxis_title='SHAP value',
                height=600,
                showlegend=False
            )

            # Rotar etiquetas del eje X si hay muchas características
            if len(self.shap_sum.index) > 10:
                fig_box.update_layout(xaxis_tickangle=-45)

            filename = f'{self.save_path}/{self.model_name}_shap_boxplot.html'
            fig_box.write_html(filename)

            if show_figure:
                fig_box.show()

            self.IT.info_print(f"Boxplot guardado: {filename}")

        except Exception as e:
            print(f"Error generando boxplot: {e}")

    def save_waterfall_plot(self, instance_idx: int = 0):
        """
        Genera gráfico waterfall para una instancia específica.

        Parameters:
        -----------
        instance_idx : int, default=0
            Índice de la instancia a explicar
        show_figure : bool, default=False
            Si mostrar el gráfico
        """
        try:
            plt.figure(figsize=(10, 8))

            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values_class_positive[instance_idx],
                    base_values=self.explainer.expected_value if not isinstance(self.explainer.expected_value, list) else self.explainer.expected_value[1],
                    data=self.x_test.iloc[instance_idx]
                ),
                show=False
            )

            filename = f'{self.save_path}/{self.model_name}_shap_waterfall_instance_{instance_idx}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            self.IT.info_print(f"Waterfall plot guardado: {filename}")

        except Exception as e:
            print(f"Error generando waterfall plot: {e}")

    def save_force_plot(self, instance_idx: int = 0):
        """
        Genera force plot para una instancia específica.

        Parameters:
        -----------
        instance_idx : int, default=0
            Índice de la instancia a explicar
        """
        try:
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]  # Para clasificación binaria

            force_plot = shap.force_plot(
                expected_value,
                self.shap_values_class_positive[instance_idx],
                self.x_test.iloc[instance_idx],
                show=False
            )

            filename = f'{self.save_path}/{self.model_name}_shap_force_instance_{instance_idx}.html'
            shap.save_html(filename, force_plot)

            self.IT.info_print(f"Force plot guardado: {filename}")

        except Exception as e:
            print(f"Error generando force plot: {e}")

    def run(self, include_waterfall: bool = True, include_force: bool = True):
        """
        Ejecuta todos los métodos de visualización.

        Parameters:
        -----------
        include_waterfall : bool, default=True
            Si incluir waterfall plot
        include_force : bool, default=True
            Si incluir force plot
        """

        # Gráficos principales
        self.plot_summary("violin")
        self.plot_summary("dot")
        self.save_barplot()
        self.save_boxplot()

        # Gráficos adicionales para instancias específicas
        if include_waterfall and len(self.x_test) > 0:
            self.save_waterfall_plot(0)

        if include_force and len(self.x_test) > 0:
            self.save_force_plot(0)

        # Guardar resumen de importancia
        summary_df = self.get_feature_importance_summary()

        summary_filename = f'{self.save_path}/{self.model_name}_shap_feature_importance_summary.csv'
        summary_df.to_csv(summary_filename, index=False)

        self.IT.info_print(f"Dataframe con las importancias relativas {summary_filename} guardado OK")

        return summary_df
