from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from _hx_model_evaluation_tools import HxFeatureImportancesandCorrelations
from _hx_model_evaluation_tools.hx_model_evaluation_graphics import HxModelEvaluationGraphics
from constants_and_tools import ConstantsAndTools
from typing import Dict, Any, List
import plotly.express as px
import pandas as pd
import numpy as np


class MulticlassClassifierMetricsCalculations:
    def __init__(self,
                 probs_result_df,
                 model_save_path: str,
                 x_test_df: pd.DataFrame,
                 model,
                 importances: list,
                 target_col_name: str,
                 model_name="a",
                 bins_dict=None,
                 n_classes: int = 2):
        """
        Este metodo se va a encargar de obtener las métricas de las target, su estratificación, llamar al hx_shap_tools si es preciso y devolver la informacion.
        Adaptado para clasificación multiclase.

        :param probs_result_df: Diccionario que contiene el y_test real, las probabilidades predichas para cada clase
        :param model_save_path: Ruta donde guardar los modelos
        :param x_test_df: DataFrame con features de test
        :param model: Modelo entrenado
        :param importances: Lista de importancias de features
        :param target_col_name: Nombre de la columna target
        :param model_name: Nombre del modelo
        :param bins_dict: Diccionario con bins para estratificación
        :param n_classes: Número de clases (por defecto 2 para binaria)
        """
        # ----------------------------------------------------------------------------------------------------
        # -- 1: Instancio toolkits que voy a necesitar
        # ----------------------------------------------------------------------------------------------------

        # ---- 1.1: Instancio constants
        self.CT: ConstantsAndTools = ConstantsAndTools()

        # ---- 1.2: Los gráficos de métricas
        self.GT: HxModelEvaluationGraphics = HxModelEvaluationGraphics()

        # ----------------------------------------------------------------------------------------------------
        # -- 2: Almaceno parámetros en propiedades
        # ----------------------------------------------------------------------------------------------------

        # ---- 2.1: Copio el df que contiene el valor real de test y las probabilidades
        self.probs_result_df: pd.DataFrame = probs_result_df.copy()

        # ---- 2.2: Obtengo el nombre de la columna target
        self.target_col_name: str = target_col_name

        # ---- 2.3: Número de clases
        self.n_classes: int = n_classes

        # ---- 2.4: Creo el dataframe de metricas para estratificacion
        self.stratify_df: pd.DataFrame = pd.DataFrame(columns=["confidence"])

        # ---- 2.5: Creamos la columna pred_value_{self.target_col_name} que contiene las predicciones del modelo (clase con mayor probabilidad)
        proba_columns = [col for col in self.probs_result_df.columns if f'proba_class_' in col]

        # Para multiclase: seleccionar la clase con mayor probabilidad
        proba_array = self.probs_result_df[proba_columns].values
        self.probs_result_df[f"pred_value_{self.target_col_name}"] = np.argmax(proba_array, axis=1)

        # ---- 2.6: Almaceno el path donde se van a guardar los informes (si procede)
        self.model_save_path: str = model_save_path

        # ---- 2.7: Almaceno el df de test sin la target en una propiedad
        self.X_test_df: pd.DataFrame = x_test_df

        # ---- 2.8: Almaceno la instancia del modelo en una propiedad
        self.model = model

        # ---- 2.9: Almaceno la lista de pesos del modelo
        self.importances: List = importances

        # ---- 2.10: Almaceno el nombre del modelo
        self.model_name: str = model_name

        # ---- 2.11: Almaceno el diccionario de bins que se va a usar para estratificar
        self.bins_dict: dict = bins_dict

    def run(self):
        """
        Metodo que realiza la evaluacion completa de un modelo
        1: Obtiene los pesos
        2: Obtiene las métricas
        3: Obtiene la estratificacion
        :return: Diccionario con métricas
        """
        # ----------------------------------------------------------------------------------------------------
        # -- 1: Realizo la evaluacion de pesos y correlaciones entre variables
        # ----------------------------------------------------------------------------------------------------

        # ---- 1.1: En caso de que existan pesos en la propiedad (Lista con los pesos) ejecutamos
        if self.importances is not None and self.importances[0] is not None:
            try:
                HxFeatureImportancesandCorrelations(self.X_test_df, self.importances[0]).run()
            except (TypeError, ValueError):
                try:
                    HxFeatureImportancesandCorrelations(self.X_test_df, self.importances).run()
                except (TypeError, ValueError):
                    pass

        # ----------------------------------------------------------------------------------------------------
        # -- 2: Ejecutamos evaluate classification para realizar todas las tareas de evaluacion y retornamos
        # ----------------------------------------------------------------------------------------------------

        # ---- 2.1: Llamo al metodo de evaluar clasificacion y obtengo las metricas
        metrics_dict: dict = self.evaluate_clasification()

        # ---- 2.2: Retorno las metricas
        return metrics_dict

    def evaluate_clasification(self):
        """
        Metodo para obtener un diccionario de métricas de cada columna target
        Adaptado para multiclase
        :return: Diccionario con métricas
        """
        # ----------------------------------------------------------------------------------------------------
        # -- 1: Defino diccionario maestro y obtengo columna de aciertos
        # ----------------------------------------------------------------------------------------------------

        # ---- 1.1: Creo el diccionario de metricas
        all_metrics_dict: dict = {}

        # ---- 1.2: Cruzo la columna del valor real con el valor predicho para obtener los aciertos del modelo
        self.probs_result_df[f"aciertos_{self.target_col_name}"] = (
                self.probs_result_df[f"real_value_{self.target_col_name}"] ==
                self.probs_result_df[f"pred_value_{self.target_col_name}"]
        )

        # ----------------------------------------------------------------------------------------------------
        # -- 2: Obtengo la matriz de confusion para multiclase
        # ----------------------------------------------------------------------------------------------------

        # ---- 2.1: Obtengo los valores reales y predichos
        y_true = self.probs_result_df[f"real_value_{self.target_col_name}"].values
        y_pred = self.probs_result_df[f"pred_value_{self.target_col_name}"].values

        # ---- 2.2: Calculo la matriz de confusión
        cm = confusion_matrix(y_true, y_pred, labels=range(self.n_classes))

        # ---- 2.3: Calculo métricas por clase
        class_metrics = {}
        for i in range(self.n_classes):
            # Verdaderos positivos para la clase i
            tp = cm[i, i]

            # Falsos positivos para la clase i (predicciones de clase i que no son i)
            fp = cm[:, i].sum() - tp

            # Falsos negativos para la clase i (reales de clase i predichos como otras clases)
            fn = cm[i, :].sum() - tp

            # Verdaderos negativos para la clase i (taquello que no es clase i correctamente identificado)
            tn = cm.sum() - (tp + fp + fn)

            class_metrics[f"class_{i}"] = {
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "accuracy": round(float((tp + tn) / (tp + tn + fp + fn)), 2) if (tp + tn + fp + fn) > 0 else 0,
                "recall": round(float(tp / (tp + fn)), 2) if (tp + fn) > 0 else 0,
                "precision": round(float(tp / (tp + fp)), 2) if (tp + fp) > 0 else 0,
                "specificity": round(float(tn / (tn + fp)), 2) if (tn + fp) > 0 else 0
            }

        # ----------------------------------------------------------------------------------------------------
        # -- 3: Calculo métricas globales
        # ----------------------------------------------------------------------------------------------------

        # ---- 3.1: Pinto la entrada en consola
        self.CT.IT.sub_intro_print(f"ClassifierMetricsCalculation 1: Métricas {self.model_name} {self.target_col_name} - {self.n_classes} clases")

        # ---- 3.2: Accuracy global
        accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)
        self.CT.IT.info_print(f"Accuracy Global:      {accuracy:.2f} %", "light_cyan")

        # ---- 3.3: Balanced Accuracy (promedio de recall por clase)
        balanced_accuracy = round(balanced_accuracy_score(y_true, y_pred) * 100, 2)
        self.CT.IT.info_print(f"Balanced Accuracy:    {balanced_accuracy:.2f} %", "light_cyan")

        # ---- 3.4: F1 Macro (promedio de F1 por clase)
        f1_macro = round(f1_score(y_true, y_pred, average='macro') * 100, 2)
        self.CT.IT.info_print(f"F1 Macro:             {f1_macro:.2f} %", "light_cyan")

        # ---- 3.5: F1 Micro (F1 global considerando todos los TP, FP, FN)
        f1_micro = round(f1_score(y_true, y_pred, average='micro') * 100, 2)
        self.CT.IT.info_print(f"F1 Micro:             {f1_micro:.2f} %", "light_cyan")

        # ---- 3.6: Precision Macro
        precision_macro = round(precision_score(y_true, y_pred, average='macro', zero_division=0) * 100, 2)
        self.CT.IT.info_print(f"Precision Macro:      {precision_macro:.2f} %", "light_cyan")

        # ---- 3.7: Recall Macro
        recall_macro = round(recall_score(y_true, y_pred, average='macro', zero_division=0) * 100, 2)
        self.CT.IT.info_print(f"Recall Macro:         {recall_macro:.2f} %", "light_cyan")

        # ----------------------------------------------------------------------------------------------------
        # -- 4: Calculo y muestro la estratificacion (usando la probabilidad de la clase predicha)
        # ----------------------------------------------------------------------------------------------------

        # ---- 4.1: Para multiclase, usamos la máxima probabilidad como confianza
        proba_columns = [col for col in self.probs_result_df.columns if f'proba_class_' in col]

        max_proba = self.probs_result_df[proba_columns].max(axis=1)
        self.probs_result_df['max_proba'] = max_proba

        # ---- 4.2: Creamos la estratificación basada en la máxima probabilidad
        self.probs_result_df[f'stratification_{self.target_col_name}'] = pd.cut(
            self.probs_result_df['max_proba'],
            bins=self.bins_dict["bins"],
            labels=self.bins_dict["labels"],
            right=False
        )

        # ---- 4.3: Creo el stratify df
        stratify_df_base_cols: list = ["confidence", "total_predicho", "aciertos", "fallos"]
        # Agregar columnas por clase
        for i in range(self.n_classes):
            stratify_df_base_cols.extend([f"class_{i}_correct", f"class_{i}_incorrect"])

        self.stratify_df: pd.DataFrame = pd.DataFrame(
            columns=[f'{z}_{self.target_col_name}' if z != "confidence" else f"{z}" for z in stratify_df_base_cols]
        )

        # ---- 4.4: Relleno iterativamente los bins de estratificacion
        total_rows = self.probs_result_df.shape[0]
        for bn in self.bins_dict["labels"][::-1]:
            if bn in self.probs_result_df[f"stratification_{self.target_col_name}"].unique().tolist():
                bn_df = self.probs_result_df[self.probs_result_df[f"stratification_{self.target_col_name}"] == bn]
                bn_df_aciertos = bn_df[bn_df[f"aciertos_{self.target_col_name}"]]
                bn_df_fallos = bn_df[~bn_df[f"aciertos_{self.target_col_name}"]]

                # Datos básicos del bin
                row_data = {
                    f"confidence": bn,
                    f"total_predicho_{self.target_col_name}": f"{round(bn_df.shape[0] / total_rows * 100, 2)}%  ({bn_df.shape[0]}/{total_rows})",
                    f"aciertos_{self.target_col_name}": f"{round(bn_df_aciertos.shape[0] / bn_df.shape[0] * 100, 2)}%  ({bn_df_aciertos.shape[0]}/{bn_df.shape[0]})" if bn_df.shape[
                                                                                                                                                                            0] != 0 else pd.NA,
                    f"fallos_{self.target_col_name}": f"{round(bn_df_fallos.shape[0] / bn_df.shape[0] * 100, 2)}%  ({bn_df_fallos.shape[0]}/{bn_df.shape[0]})" if bn_df.shape[
                                                                                                                                                                      0] != 0 else pd.NA,
                }

                # Datos por clase
                for i in range(self.n_classes):
                    class_correct = bn_df_aciertos[bn_df_aciertos[f"pred_value_{self.target_col_name}"] == i].shape[0]
                    class_incorrect = bn_df_fallos[bn_df_fallos[f"real_value_{self.target_col_name}"] == i].shape[0]

                    row_data[f"class_{i}_correct_{self.target_col_name}"] = f"{class_correct}"
                    row_data[f"class_{i}_incorrect_{self.target_col_name}"] = f"{class_incorrect}"

                self.stratify_df.loc[len(self.stratify_df)] = row_data

        # ----------------------------------------------------------------------------------------------------
        # -- 5: Creo gráficos, pinto y retorno
        # ----------------------------------------------------------------------------------------------------

        # ---- 5.1: Creo el gráfico de estratificacion en barras (adaptado para multiclase)
        self.plot_stratify_barplot_multiclass(
            self.probs_result_df,
            f"{self.model_save_path}/stratify_barplot_{self.target_col_name}.html",
            f"stratification_{self.target_col_name}",
            f"aciertos_{self.target_col_name}",
            f"pred_value_{self.target_col_name}",
            self.n_classes
        )

        # ---- 5.2: Creo el diccionario con todas las metricas
        all_metrics_dict[f"{self.target_col_name}"] = {
            "metrics": {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "confusion_matrix": cm.tolist(),
                "class_metrics": class_metrics
            }
        }

        # ---- 5.3: Pinto la entrada y las metricas de estratificacion
        self.CT.IT.sub_intro_print(f"ClassifierMetricsCalculation 2: Tabla de estratificacion {self.model_name}")
        self.CT.IT.print_tabulate_df(self.stratify_df, row_print=100)

        # ---- 5.4: Almaceno el gráfico de la estratificacion en formato tabla
        self.GT.plot_dataframe_html(self.stratify_df, save_path=f"{self.model_save_path}/stratify_df.html", save=True)

        # ---- 5.5: Retorno el diccionario completo de metricas
        return all_metrics_dict

    # <editor-fold desc="Metodos secundarios usados en evaluateClasification    --------------------------------------------------------------------------------------------------">

    @staticmethod
    def plot_stratify_barplot_multiclass(df: pd.DataFrame, path_and_name: str, stratification_colname: str,
                                         accurate_colname: str, pred_class_colname: str, n_classes: int, show: bool = False):
        """
        Gráfico de barras para multiclase
        """
        # Calcular conteos por clase y estratificación
        df['count'] = 1
        results = []

        for strat in df[stratification_colname].unique():
            for class_idx in range(n_classes):
                strat_class_df = df[(df[stratification_colname] == strat) & (df[pred_class_colname] == class_idx)]
                correct = strat_class_df[accurate_colname].sum()
                incorrect = len(strat_class_df) - correct

                results.append({
                    'stratification': strat,
                    'class': f'Class {class_idx}',
                    'correct': correct,
                    'incorrect': incorrect
                })

        results_df = pd.DataFrame(results)

        # Crear gráfico apilado
        fig = px.bar(results_df, x='stratification', y=['correct', 'incorrect'],
                     color_discrete_map={'correct': '#3182bd', 'incorrect': '#de2d26'},
                     title='Distribución de aciertos por clase y rango de confianza',
                     labels={'stratification': 'Rango de Confianza', 'value': 'Cantidad', 'variable': 'Resultado'},
                     barmode='stack')

        fig.update_layout(title={'text': 'Distribución de aciertos por clase y rango de confianza',
                                 'x': 0.5, 'y': 0.95,
                                 'font': {'size': 20, 'color': 'black'},
                                 'pad': {'b': 30}})

        fig.write_html(path_and_name)
        if show:
            fig.show()

    # </editor-fold>


class EvaluateMulticlassClassifier:
    def __init__(self, data_dict: dict, model_name: str, selected_metric: str | None = None, n_classes: int = 2):

        # --------------------------------------------------------------------------------------------
        # -- 1: Instancio clases de toolkits y almaceno en propiedades
        # --------------------------------------------------------------------------------------------

        # ---- 1.1: Instancio constants
        self.CT: ConstantsAndTools = ConstantsAndTools()

        # ---- 1.2: Almaceno el nombre del modelo
        self.model_name: str = model_name

        # ---- 1.3: Almaceno la metrica seleccionada
        self.selected_metric: str = selected_metric

        # ---- 1.4: Almaceno el número de clases
        self.n_classes: int = n_classes

        # ---- 1.5: Almaceno los valores reales del target
        self.y_train = data_dict["y_train"]
        self.y_eval = data_dict["y_eval"]
        self.y_test = data_dict["y_test"]

        # ---- 1.6: Almaceno los valores predichos del target
        self.y_pred_train = data_dict["y_pred_train"]
        self.y_pred_eval = data_dict["y_pred_eval"]
        self.y_pred_test = data_dict["y_pred_test"]

        # ---- 1.7: Almaceno las columnas target
        self.target_col_name: list = data_dict["target_col_name"]

        # ---- 1.8: Inicializo en None el df que va a contener las métricas
        self.metrics_df = None

    def calculate_and_print_metrics(self):
        """
        Calcula métricas para clasificación multiclase
        :return: Diccionario con métricas
        """
        # --------------------------------------------------------------------------------------------
        # -- 1: Pinto y calculo las metricas
        # --------------------------------------------------------------------------------------------

        # ---- 1.1: Pinto la entrada
        self.CT.IT.sub_intro_print("EvaluateClassifier: Calculando metricas de entrenamiento (Multiclase)...")

        # --------------------------------------------------------------------------------------------
        # -- 2: Calculo métricas para multiclase
        # --------------------------------------------------------------------------------------------

        # ---- 2.1: Accuracy
        accuracy_score_train = accuracy_score(self.y_train, self.y_pred_train)
        accuracy_score_eval = accuracy_score(self.y_eval, self.y_pred_eval) if self.y_eval is not None else None
        accuracy_score_test = accuracy_score(self.y_test, self.y_pred_test)

        # ---- 2.2: Balanced Accuracy
        balanced_acc_score_train = balanced_accuracy_score(self.y_train, self.y_pred_train)
        balanced_acc_score_eval = balanced_accuracy_score(self.y_eval, self.y_pred_eval) if self.y_eval is not None else None
        balanced_acc_score_test = balanced_accuracy_score(self.y_test, self.y_pred_test)

        # ---- 2.3: F1 Scores
        f1_macro_train = f1_score(self.y_train, self.y_pred_train, average='macro')
        f1_macro_eval = f1_score(self.y_eval, self.y_pred_eval, average='macro') if self.y_eval is not None else None
        f1_macro_test = f1_score(self.y_test, self.y_pred_test, average='macro')

        f1_micro_train = f1_score(self.y_train, self.y_pred_train, average='micro')
        f1_micro_eval = f1_score(self.y_eval, self.y_pred_eval, average='micro') if self.y_eval is not None else None
        f1_micro_test = f1_score(self.y_test, self.y_pred_test, average='micro')

        f1_weighted_train = f1_score(self.y_train, self.y_pred_train, average='weighted')
        f1_weighted_eval = f1_score(self.y_eval, self.y_pred_eval, average='weighted') if self.y_eval is not None else None
        f1_weighted_test = f1_score(self.y_test, self.y_pred_test, average='weighted')

        # ---- 2.4: Precision Scores
        precision_macro_train = precision_score(self.y_train, self.y_pred_train, average='macro', zero_division=0)
        precision_macro_eval = precision_score(self.y_eval, self.y_pred_eval, average='macro', zero_division=0) if self.y_eval is not None else None
        precision_macro_test = precision_score(self.y_test, self.y_pred_test, average='macro', zero_division=0)

        precision_micro_train = precision_score(self.y_train, self.y_pred_train, average='micro', zero_division=0)
        precision_micro_eval = precision_score(self.y_eval, self.y_pred_eval, average='micro', zero_division=0) if self.y_eval is not None else None
        precision_micro_test = precision_score(self.y_test, self.y_pred_test, average='micro', zero_division=0)

        # ---- 2.5: Recall Scores
        recall_macro_train = recall_score(self.y_train, self.y_pred_train, average='macro', zero_division=0)
        recall_macro_eval = recall_score(self.y_eval, self.y_pred_eval, average='macro', zero_division=0) if self.y_eval is not None else None
        recall_macro_test = recall_score(self.y_test, self.y_pred_test, average='macro', zero_division=0)

        recall_micro_train = recall_score(self.y_train, self.y_pred_train, average='micro', zero_division=0)
        recall_micro_eval = recall_score(self.y_eval, self.y_pred_eval, average='micro', zero_division=0) if self.y_eval is not None else None
        recall_micro_test = recall_score(self.y_test, self.y_pred_test, average='micro', zero_division=0)

        # ---- 2.6: ROC AUC  (one-vs-rest para multiclase)

        y_train_bin = label_binarize(self.y_train, classes=range(self.n_classes))
        y_test_bin = label_binarize(self.y_test, classes=range(self.n_classes))

        roc_auc_train = roc_auc_score(y_train_bin, label_binarize(self.y_pred_train, classes=range(self.n_classes)), average='macro', multi_class='ovr')
        roc_auc_test = roc_auc_score(y_test_bin, label_binarize(self.y_pred_test, classes=range(self.n_classes)), average='macro', multi_class='ovr')
        roc_auc_eval = roc_auc_score(label_binarize(self.y_eval, classes=range(self.n_classes)),
                                     label_binarize(self.y_pred_eval, classes=range(self.n_classes)),
                                     average='macro', multi_class='ovr') if self.y_eval is not None else None

        # ---- 2.7: Matrices de confusión
        confusion_matrix_train = confusion_matrix(self.y_train, self.y_pred_train)
        confusion_matrix_eval = confusion_matrix(self.y_eval, self.y_pred_eval) if self.y_eval is not None else None
        confusion_matrix_test = confusion_matrix(self.y_test, self.y_pred_test)

        # --------------------------------------------------------------------------------------------
        # -- 3: Creo el dataframe de métricas
        # --------------------------------------------------------------------------------------------

        hx_round = lambda x: round(x, 4) if x is not None else pd.NA

        self.metrics_df: pd.DataFrame = pd.DataFrame(columns=("metric", f"train_{self.target_col_name}", f"eval_{self.target_col_name}", f"test_{self.target_col_name}"))

        # Agregar todas las métricas
        metrics_data = [
            ("accuracy", hx_round(accuracy_score_train), hx_round(accuracy_score_eval), hx_round(accuracy_score_test)),
            ("balanced_accuracy", hx_round(balanced_acc_score_train), hx_round(balanced_acc_score_eval), hx_round(balanced_acc_score_test)),
            ("f1_macro", hx_round(f1_macro_train), hx_round(f1_macro_eval), hx_round(f1_macro_test)),
            ("f1_micro", hx_round(f1_micro_train), hx_round(f1_micro_eval), hx_round(f1_micro_test)),
            ("f1_weighted", hx_round(f1_weighted_train), hx_round(f1_weighted_eval), hx_round(f1_weighted_test)),
            ("precision_macro", hx_round(precision_macro_train), hx_round(precision_macro_eval), hx_round(precision_macro_test)),
            ("precision_micro", hx_round(precision_micro_train), hx_round(precision_micro_eval), hx_round(precision_micro_test)),
            ("recall_macro", hx_round(recall_macro_train), hx_round(recall_macro_eval), hx_round(recall_macro_test)),
            ("recall_micro", hx_round(recall_micro_train), hx_round(recall_micro_eval), hx_round(recall_micro_test)),
            ("roc_auc_ovr", hx_round(roc_auc_train), hx_round(roc_auc_eval), hx_round(roc_auc_test))
        ]

        for metric_name, train_val, eval_val, test_val in metrics_data:
            self.metrics_df.loc[len(self.metrics_df)] = [metric_name, train_val, eval_val, test_val]

        # --------------------------------------------------------------------------------------------
        # -- 4: Proceso y transformo el metrics_df
        # --------------------------------------------------------------------------------------------

        # ---- 4.1: Elimino columnas que contengan únicamente valores NaN
        self.metrics_df = self.metrics_df.dropna(axis=1, how="all")

        # ---- 4.2: Convertir a float64 todas las columnas excepto 'metric'
        self.metrics_df.loc[:, self.metrics_df.columns != 'metric'] = self.metrics_df.loc[:, self.metrics_df.columns != 'metric'].astype('float64')

        # --------------------------------------------------------------------------------------------
        # -- 5: Pinto el df y genero el diccionario de métricas
        # --------------------------------------------------------------------------------------------

        # ---- 5.1: Pinto el df
        self.CT.IT.print_tabulate_df(self.metrics_df)

        # ---- 5.2: Creo el diccionario de metricas
        gen_metric_dict: dict = {}
        for metric in self.metrics_df["metric"].unique():
            gen_metric_dict[f"{metric}_train"] = self.metrics_df[self.metrics_df["metric"] == metric][f"train_{self.target_col_name}"].iloc[0]
            gen_metric_dict[f"{metric}_test"] = self.metrics_df[self.metrics_df["metric"] == metric][f"test_{self.target_col_name}"].iloc[0]
            if f"eval_{self.target_col_name}" in self.metrics_df.columns:
                gen_metric_dict[f"{metric}_eval"] = self.metrics_df[self.metrics_df["metric"] == metric][f"eval_{self.target_col_name}"].iloc[0]

        # Agregar matrices de confusión
        gen_metric_dict["confusion_matrix_train"] = confusion_matrix_train.tolist()
        gen_metric_dict["confusion_matrix_test"] = confusion_matrix_test.tolist()
        if confusion_matrix_eval is not None:
            gen_metric_dict["confusion_matrix_eval"] = confusion_matrix_eval.tolist()

        del self.metrics_df

        # ---- 5.3: Devuelvo el diccionario con la metrica seleccionada y las metricas totales
        result: Dict[str, Any] = {}

        if self.selected_metric is not None:
            metric_map = {
                "accuracy": ("accuracy", "accuracy"),
                "balanced_accuracy": ("balanced_accuracy", "balanced_accuracy"),
                "f1_macro": ("f1_macro", "f1_macro"),
                "f1_micro": ("f1_micro", "f1_micro"),
                "f1_weighted": ("f1_weighted", "f1_weighted"),
                "precision_macro": ("precision_macro", "precision_macro"),
                "recall_macro": ("recall_macro", "recall_macro"),
                "roc_auc": ("roc_auc_ovr", "roc_auc")
            }

            if self.selected_metric in metric_map:
                metric_key, display_name = metric_map[self.selected_metric]
                result = {
                    "metric_train": gen_metric_dict.get(f"{metric_key}_train"),
                    "metric_eval": gen_metric_dict.get(f"{metric_key}_eval"),
                    "metric_test": gen_metric_dict.get(f"{metric_key}_test"),
                    "all_metrics": gen_metric_dict,
                    "selected_metric": display_name
                }
            else:
                raise ValueError(f"La métrica seleccionada '{self.selected_metric}' no está contemplada para multiclase")

            return result

        return None