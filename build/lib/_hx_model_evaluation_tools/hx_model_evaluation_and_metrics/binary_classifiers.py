from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from _hx_model_evaluation_tools import HxFeatureImportancesandCorrelations
from _hx_model_evaluation_tools.hx_model_evaluation_graphics import HxModelEvaluationGraphics
from constants_and_tools import ConstantsAndTools
from typing import Dict, Any, List
import plotly.express as px
import pandas as pd
import numpy as np


class BinaryClassifierMetricsCalculations:
    def __init__(self,
                 probs_result_df,
                 model_save_path: str,
                 x_test_df: pd.DataFrame,
                 model,
                 importances: list,
                 target_col_name: str,
                 model_name="a",
                 bins_dict=None):
        """
        Este metodo se va a encargar de obtener las métricas de las target, su estratificación, llamar al hx_shap_tools si es preciso y devolver la informacion.
        Claves: Si se pasan las listas real_value_list, pos_prob_list, o pred_val_list con info, se usará su contenido. Sino, poe defecto se usarán los nombres que
        se han definido en las propiedades del constructor

        :param probs_result_df: Diccionario que contiene el y_test real, la probabilidad positiva predicha y la probabilidad negativa predicha
        :param model_save_path:
        :param x_test_df:
        :param model:
        :param importances:
        :param target_col_name:
        :param model_name:
        :param bins_dict:
        """

        # ----------------------------------------------------------------------------------------------------
        # -- 1: Instancio toolkits que voy a nacesitar
        # ----------------------------------------------------------------------------------------------------

        # ---- 1.1: Instancio constants
        self.CT: ConstantsAndTools = ConstantsAndTools()

        # ---- 1.2: Los gráficos de métricas
        self.GT: HxModelEvaluationGraphics = HxModelEvaluationGraphics()

        # ----------------------------------------------------------------------------------------------------
        # -- 2: Almaceno parámetros en propiedades
        # ----------------------------------------------------------------------------------------------------

        # ---- 2.1: Copio el df que contiene el valor real de test, la probabilidad positiva y la negativa
        self.probs_result_df: pd.DataFrame = probs_result_df.copy()

        # ---- 2.2: Obtengo el nombre de la columna target
        self.target_col_name: str = target_col_name

        # ---- 2.3: Creo el dataframe de metricas para estratificacion
        self.stratify_df: pd.DataFrame = pd.DataFrame(columns=["confidence"])

        # ---- 2.4: Creamos la columna pred_value_{self.target_col_name} que contiene las prediciones del modelo redondeadas
        self.probs_result_df[f"pred_value_{self.target_col_name}"] = np.round(self.probs_result_df[f'positive_proba_{self.target_col_name}'])

        # ---- 2.5: Almaceno el path donde se van a guardar los informes (si procede)
        self.model_save_path: str = model_save_path

        # ---- 2.6: Almaceno el df de test sin la target en una propiedad
        self.X_test_df: pd.DataFrame = x_test_df

        # ---- 2.7: Almaceno la instancia del modelo en una propiedad
        self.model = model

        # ---- 2.8: Almaceno la lista de pesos del modelo
        self.importances: List = importances

        # ---- 2.9: Almaceno el nombre del modelo
        self.model_name: str = model_name

        # ---- 2.10: Almaceno el diccionario de bins que se va a usar para estratificar
        self.bins_dict: dict = bins_dict

    def run(self):
        """
        Metodo que realiza la evaluacion completa de un modelo
        1: Obtiene los pesos
        2: Obtiene las métricas
        3: Obtiene la estratificacion
        :return:
        """
        # ----------------------------------------------------------------------------------------------------
        # -- 1: Realizo la evaluacion de pesos y correlaciones entre variables
        # ----------------------------------------------------------------------------------------------------

        # ---- 1.1: En caso de que existan pesos en la propiedad (Lista con los pesos) ejecutamis
        if self.importances is not None and self.importances[0] is not None:

            try:
                HxFeatureImportancesandCorrelations(self.X_test_df, self.importances[0]).run()
            except TypeError:
                try:
                    HxFeatureImportancesandCorrelations(self.X_test_df, self.importances).run()
                except TypeError:
                    pass
                except ValueError:
                    pass
            except ValueError:
                pass

        # ----------------------------------------------------------------------------------------------------
        # -- 2: Ejecutamos evaluate classification para realizar todas las tareas de evaluacion y retornamos
        # ----------------------------------------------------------------------------------------------------

        # ---- 2.1: Llamo al metodo de evaluar clasificacion y obtengo las metricas
        metrics_dict: dict = self.evaluate_clasification()

        # ---- 2.2: Retorno las metricas (no lo hago directo porque me facilita el debug si es necesario)
        return metrics_dict

    def evaluate_clasification(self):
        """
        Metodo para obtener un diccionario de métricas de cada columna target
        :return:
        """
        # ----------------------------------------------------------------------------------------------------
        # -- 1: Defino diccionario maestro y obtengo columna de aciertos
        # ----------------------------------------------------------------------------------------------------

        # ---- 1.1: Creo el diccionario de metricas
        all_metrics_dict: dict = {}

        # ---- 1.2: Cruzo la columna de la probabilidad positiva con el valor real (y_test) para obtener los aciertos del modelo
        self.probs_result_df[f"aciertos_{self.target_col_name}"] = np.where(
            (self.probs_result_df[f"real_value_{self.target_col_name}"] == 1) & (self.probs_result_df[f"positive_proba_{self.target_col_name}"] >= 0.5) |
            (self.probs_result_df[f"real_value_{self.target_col_name}"] == 0) & (self.probs_result_df[f"positive_proba_{self.target_col_name}"] < 0.5),
            True,  # Acierto si ambas condiciones son ciertas
            False  # Fallo si no coinciden
        )

        # ----------------------------------------------------------------------------------------------------
        # -- 2: Obtengo a mano la matriz de confusion (necesito mantenerlos en flotante para el stratify)
        # ----------------------------------------------------------------------------------------------------

        # ---- 2.1: Obtengo un dataframe con los Aciertos: las filas donde el acierto es True
        df_true: pd.DataFrame = self.probs_result_df[self.probs_result_df[f"aciertos_{self.target_col_name}"]]

        # ---- 2.2: Obtengo los VP: El modelo predice si y es si (VERDADEROS POSITIVOS)
        true_positive = self.probs_result_df[(self.probs_result_df[f"real_value_{self.target_col_name}"] == 1.0) &
                                             (self.probs_result_df[f"aciertos_{self.target_col_name}"])].shape[0]

        # ---- 2.3: Obtengo los VN: El modelo predice no y es no (VERDADEROS NEGATIVOS)
        true_negative = self.probs_result_df[(self.probs_result_df[f"real_value_{self.target_col_name}"] == 0.0) &
                                             (self.probs_result_df[f"aciertos_{self.target_col_name}"])].shape[0]

        # ---- 2.4: Obtengo los FN: El modelo predice no y en realidad es si (FALSOS NEGATIVOS)
        false_negative = self.probs_result_df[(self.probs_result_df[f"real_value_{self.target_col_name}"] == 1.0) &
                                              (~self.probs_result_df[f"aciertos_{self.target_col_name}"])].shape[0]

        # ---- 2.5: Obtengo los FP: El modelo predice si y en realidad es no (FALSOS POSITIVOS)
        false_positive = self.probs_result_df[(self.probs_result_df[f"real_value_{self.target_col_name}"] == 0.0) &
                                              (~self.probs_result_df[f"aciertos_{self.target_col_name}"])].shape[0]

        # ----------------------------------------------------------------------------------------------------
        # -- 3: Calculo y muestro las métricas
        # ----------------------------------------------------------------------------------------------------

        # ---- 3.1: Pinto la entrada en consola
        self.CT.IT.sub_intro_print(f"ClassifierMetricsCalculation 1: Métricas {self.model_name} {self.target_col_name}")

        # ---- 3.2: Accuracy
        accuracy = round(df_true.shape[0] / self.probs_result_df.shape[0] * 100, 3)
        self.CT.IT.info_print(f"Accuracy:      {accuracy} % -------  Formula: ({true_positive} + {true_negative})  / ({false_negative} + {false_negative}) * 100",
                              "light_cyan")

        # ---- 3.3: Recall (SENSIBILIDAD)
        sensibility = round(true_positive / (true_positive + false_negative) * 100, 3)
        self.CT.IT.info_print(f"Recall:        {sensibility} % -------  Formula: {true_positive} / ({true_positive} + {false_negative}) * 100",
                              "light_cyan")

        # ---- 3.4: Specificity: (ESPECIFICIDAD)
        specificity = round(true_negative / (true_negative + false_positive) * 100, 3)
        self.CT.IT.info_print(f"Specificity:   {specificity} % -------  Formula: {true_negative} / ({true_negative} + {false_positive}) * 100",
                              "light_cyan")

        # ---- 3.5: F1 (F1)
        f1 = round(((accuracy * sensibility) / (accuracy + sensibility)) * 2, 3)
        self.CT.IT.info_print(f"F1:            {f1} % -------  Formula: ({accuracy * sensibility:.2f} / {accuracy + sensibility:.2f}) * 2",
                              "light_cyan")

        # ----------------------------------------------------------------------------------------------------
        # -- 4: Calculo y muestro la estratificacion
        # ----------------------------------------------------------------------------------------------------

        # ---- 4.1: Creamos la nueva columna utilizando pd.cut(), esto depende de lo que le pasamos con parametro en el bins_dict
        self.probs_result_df[f'stratification_{self.target_col_name}'] = pd.cut(self.probs_result_df[f"positive_proba_{self.target_col_name}"], bins=self.bins_dict["bins"], labels=self.bins_dict["labels"], right=False)

        # ---- 4.2: Creo el stratify df que se va a ir agregando al self.stratify df
        stratify_df_base_cols: list = ["confidence", "total_predicho", "aciertos", "fallos", "TP_total_TP", "FP_total_FP", "TN_total_TN", "FN_total_FN"]
        self.stratify_df: pd.DataFrame = pd.DataFrame(columns=[f'{z}_{self.target_col_name}' if z != "confidence" else f"{z}" for z in stratify_df_base_cols])

        # ---- 4.3: Relleno  iterativamente los bins de estratificacion en base al self.bins_dict
        for bn in self.bins_dict["labels"][::-1]:
            if bn in self.probs_result_df[f"stratification_{self.target_col_name}"].unique().tolist():
                res_shape: int = self.probs_result_df.shape[0]
                bn_df = self.probs_result_df[self.probs_result_df[f"stratification_{self.target_col_name}"] == bn]
                bn_df_aciertos = bn_df[bn_df[f"aciertos_{self.target_col_name}"]]
                bn_df_fallos = bn_df[(~bn_df[f"aciertos_{self.target_col_name}"])]
                tp_df = bn_df_aciertos[(bn_df_aciertos[f'real_value_{self.target_col_name}'] == 1.0) | (bn_df_aciertos[f'real_value_{self.target_col_name}'] == 1)]
                fp_df = bn_df_fallos[(bn_df_fallos[f'real_value_{self.target_col_name}'] == 0.0) | (bn_df_fallos[f'real_value_{self.target_col_name}'] == 0)]
                tn_df = bn_df_aciertos[(bn_df_aciertos[f'real_value_{self.target_col_name}'] == 0.0) | (bn_df_aciertos[f'real_value_{self.target_col_name}'] == 0)]
                fn_df = bn_df_fallos[(bn_df_fallos[f'real_value_{self.target_col_name}'] == 1.0) | (bn_df_fallos[f'real_value_{self.target_col_name}'] == 1)]

                self.stratify_df.loc[len(self.stratify_df)] = {
                    f"confidence": bn,
                    f"total_predicho_{self.target_col_name}": f"{round(bn_df.shape[0] / res_shape * 100, 2)}%  ({bn_df.shape[0]}/{res_shape})" if res_shape != 0 else pd.NA,
                    f"aciertos_{self.target_col_name}": f"{round(bn_df_aciertos.shape[0] / bn_df.shape[0] * 100, 2)}%  ({bn_df_aciertos.shape[0]}/{bn_df.shape[0]})" if bn_df.shape[0] != 0 else pd.NA,
                    f"fallos_{self.target_col_name}": f"{round(bn_df_fallos.shape[0] / bn_df.shape[0] * 100, 2)}%  ({bn_df_fallos.shape[0]}/{bn_df.shape[0]})" if bn_df.shape[0] != 0 else pd.NA,
                    f"TP_total_TP_{self.target_col_name}": f"{round(tp_df.shape[0] / true_positive * 100, 2)}%  ({tp_df.shape[0]}/{true_positive})" if true_positive != 0 else pd.NA,
                    f"FP_total_FP_{self.target_col_name}": f"{round(fp_df.shape[0] / false_positive * 100, 2)}%  ({fp_df.shape[0]}/{false_positive})" if false_positive != 0 else pd.NA,
                    f"TN_total_TN_{self.target_col_name}": f"{round(tn_df.shape[0] / true_negative * 100, 2)}%  ({tn_df.shape[0]}/{true_negative})" if true_negative != 0 else pd.NA,
                    f"FN_total_FN_{self.target_col_name}": f"{round(fn_df.shape[0] / false_negative * 100, 2)}%  ({fn_df.shape[0]}/{false_negative})" if false_negative != 0 else pd.NA,

                }

        # ----------------------------------------------------------------------------------------------------
        # -- 5: Creo gráficos, pinto y retorno
        # ----------------------------------------------------------------------------------------------------

        # ---- 5.1: Creo el gráfico de estratificacion en barras
        self.plot_stratify_barplot(self.probs_result_df, f"{self.model_save_path}/stratify_barplot_{self.target_col_name}.html",
                                   f"stratification_{self.target_col_name}", f"aciertos_{self.target_col_name}")

        # ---- 5.2: Creo el diccionario con todas las metricas que voy a retornar
        all_metrics_dict[f"{self.target_col_name}"] = {"metrics": {"accuracy": accuracy,
                                                     "recall": sensibility,
                                                     "specifity": specificity,
                                                     "f1": f1,
                                                     "ALL": {"TP": true_positive, "FP": false_positive,
                                                             "TN": true_negative, "FN": false_negative},
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
    def plot_stratify_barplot(df: pd.DataFrame, path_and_name: str, stratification_colname: str = "stratification", accurate_colname: str = "aciertos", show: bool = False):
        # Calcular acumulados
        df['count'] = 1
        df['acumulado'] = df.groupby(stratification_colname, observed=False)['count'].cumsum()

        # Calcular porcentajes de aciertos y fallos por rango
        df['porcentaje_aciertos'] = df.groupby(stratification_colname, observed=False)[accurate_colname].transform(lambda x: x.mean() * 100)
        df['porcentaje_fallos'] = 100 - df['porcentaje_aciertos']

        # Crear el gráfico con Plotly
        fig = px.bar(df.groupby([stratification_colname, accurate_colname], observed=False).agg({'count': 'sum'}).reset_index(),
                     x=stratification_colname, y='count', color=accurate_colname,
                     title='Cantidad de filas por rango y acierto',
                     labels={stratification_colname: 'Rangos', accurate_colname: 'Acierto/Fallo'},
                     color_discrete_map={1: '#3182bd', 0: '#de2d26'},  # Colores elegantes
                     barmode='stack',  # Apilar los colores dentro de cada rango
                     category_orders={accurate_colname: [1, 0]})  # Invertir orden de las categorías

        # Personalizar diseño del gráfico
        fig.update_layout(title={'text': 'Cantidad de filas por rango y acierto',
                                 'x': 0.5, 'y': 0.95,  # Alineación central del título
                                 'font': {'size': 20, 'color': 'black'},  # Tamaño y color de la fuente del título
                                 'pad': {'b': 30}},  # Espaciado inferior del título
                          xaxis_title='Rangos',
                          yaxis_title='Cantidad',
                          legend_title='Acierto/Fallo',
                          font=dict(size=12),  # Tamaño de la fuente de los ejes
                          hoverlabel=dict(font_size=14, font_family="Arial"),  # Tamaño y fuente del hover
                          hovermode="closest")  # Hover solo sobre el eje x

        # Calcular total por rango (inside update_traces)
        total_por_rango = df.groupby(stratification_colname, observed=False)['count'].sum()
        df = df.merge(total_por_rango.rename('total_por_rango'), left_on=stratification_colname, right_index=True)

        # Actualizar el tooltip con el total fijo
        fig.update_traces(hovertemplate='<b>Rango</b>: %{x}<br>' +
                                        '<b>Cantidad</b>: %{y} de ' +
                                        f'{df.shape[0]} totales<br>')

        fig.write_html(f"{path_and_name}")
        if show:
            # Mostrar el gráfico
            fig.show()

    # </editor-fold>


class EvaluateBinaryClassifier:
    def __init__(self, data_dict: dict, model_name: str, selected_metric: str | None = None):

        # --------------------------------------------------------------------------------------------
        # -- 1: Instancio clases de toolkits y almaceno en propiedades
        # --------------------------------------------------------------------------------------------

        # ---- 1.1: Instancio constants
        self.CT: ConstantsAndTools = ConstantsAndTools()

        # ---- 1.2: Almaceno el nombre del modelo
        self.model_name: str = model_name

        # ---- 1.3: Almaceno la metrica selecconada
        self.selected_metric: str = selected_metric

        # ---- 1.4: Almaceno los valores reales del target
        self.y_train = data_dict["y_train"]
        self.y_eval = data_dict["y_eval"]
        self.y_test = data_dict["y_test"]

        # ---- 1.5: Almaceno los valores predichos del target
        self.y_pred_train = data_dict["y_pred_train"]
        self.y_pred_eval = data_dict["y_pred_eval"]
        self.y_pred_test = data_dict["y_pred_test"]

        # ---- 1.6: Almaceno las columnas target en la propiedad (en esta versión, el multitarget no está habilitado, pero dejo la estructura)
        self.target_col_name: list = data_dict["target_col_name"]

        # ---- 1.7: Inicializo en None el df que va a contener las métricas
        self.metrics_df = None

    def calculate_and_print_metrics(self):

        # --------------------------------------------------------------------------------------------
        # -- 1: Pinto y calculo las metricas
        # --------------------------------------------------------------------------------------------

        # ---- 1.1: Pinto la entrada
        self.CT.IT.sub_intro_print("EvaluateBinaryClassifier: Calculando metricas de entrenamiento...")

        # ---- 1.2: Itero por las target_col, calculo metricas y agrego al df. (Actualmente solo hay una target)

        # -- Accuracy
        accuracy_score_train = accuracy_score(self.y_train, self.y_pred_train)
        accuracy_score_eval = accuracy_score(self.y_eval, self.y_pred_eval) if self.y_eval is not None else None
        accuracy_score_test = accuracy_score(self.y_test, self.y_pred_test)

        # -- Puntuacion de precision promedio
        avg_precision_score_train = average_precision_score(self.y_train, self.y_pred_train)
        avg_precision_score_eval = average_precision_score(self.y_eval, self.y_pred_eval) if self.y_eval is not None else None
        avg_precision_score_test = average_precision_score(self.y_test, self.y_pred_test)

        # -- Puntuacion de precision balanceada
        balanced_acc_score_train = balanced_accuracy_score(self.y_train, self.y_pred_train)
        balanced_acc_score_eval = balanced_accuracy_score(self.y_eval, self.y_pred_eval) if self.y_eval is not None else None
        balanced_acc_score_test = balanced_accuracy_score(self.y_test, self.y_pred_test)

        # -- Valor f1 (equilibrio entre accuracy y recall)
        confusion_matrix_train = confusion_matrix(self.y_train, self.y_pred_train)
        confusion_matrix_eval = confusion_matrix(self.y_eval, self.y_pred_eval) if self.y_eval is not None else None
        confusion_matrix_test = confusion_matrix(self.y_test, self.y_pred_test)
        specificity_train = confusion_matrix_train[0, 0] / (confusion_matrix_train[0, 0] + confusion_matrix_train[0, 1])
        specificity_eval = confusion_matrix_eval[0, 0] / (confusion_matrix_eval[0, 0] + confusion_matrix_eval[0, 1]) if self.y_eval is not None else None
        specificity_test = confusion_matrix_test[0, 0] / (confusion_matrix_test[0, 0] + confusion_matrix_test[0, 1])

        # -- Valor f1 (equilibrio entre accuracy y recall)
        f1_score_train = f1_score(self.y_train, self.y_pred_train)
        f1_score_eval = f1_score(self.y_eval, self.y_pred_eval) if self.y_eval is not None else None
        f1_score_test = f1_score(self.y_test, self.y_pred_test)

        # -- Numero de verdaderos positivos / total de positivos predichos (Zero division es para poner un valor cuando no se predicen muestras en una clase)
        precision_score_train = precision_score(self.y_train, self.y_pred_train, zero_division=1)
        precision_score_eval = precision_score(self.y_eval, self.y_pred_eval, zero_division=1) if self.y_eval is not None else None
        precision_score_test = precision_score(self.y_test, self.y_pred_test, zero_division=1)

        # -- Numero de verdaderos positivos / total de positivos reales
        recall_score_train = recall_score(self.y_train, self.y_pred_train)
        recall_score_eval = recall_score(self.y_eval, self.y_pred_eval) if self.y_eval is not None else None
        recall_score_test = recall_score(self.y_test, self.y_pred_test)

        # -- Area bajo la curva roc
        roc_auc_score_train = roc_auc_score(self.y_train, self.y_pred_train)
        roc_auc_score_eval = roc_auc_score(self.y_eval, self.y_pred_eval) if self.y_eval is not None else None
        roc_auc_score_test = roc_auc_score(self.y_test, self.y_pred_test)

        # -- Creo el df de metricas del individuo concreto
        hx_round = lambda x: round(x, 4) if x is not None else pd.NA
        self.metrics_df: pd.DataFrame = pd.DataFrame(columns=("metric", f"train_{self.target_col_name}", f"eval_{self.target_col_name}", f"test_{self.target_col_name}"))
        self.metrics_df.loc[len(self.metrics_df)] = ["accuracy", f"{hx_round(accuracy_score_train)}", hx_round(accuracy_score_eval), hx_round(accuracy_score_test)]
        self.metrics_df.loc[len(self.metrics_df)] = ["recall", f"{hx_round(recall_score_train)}", hx_round(recall_score_eval), hx_round(recall_score_test)]
        self.metrics_df.loc[len(self.metrics_df)] = ["specificity", f"{hx_round(specificity_train)}", hx_round(specificity_eval), hx_round(specificity_test)]
        self.metrics_df.loc[len(self.metrics_df)] = ["f1", f"{hx_round(f1_score_train)}", hx_round(f1_score_eval), hx_round(f1_score_test)]
        self.metrics_df.loc[len(self.metrics_df)] = ["roc_auc", f"{hx_round(roc_auc_score_train)}", hx_round(roc_auc_score_eval), hx_round(roc_auc_score_test)]
        self.metrics_df.loc[len(self.metrics_df)] = ["average_precision", f"{hx_round(avg_precision_score_train)}", hx_round(avg_precision_score_eval), hx_round(avg_precision_score_test)]
        self.metrics_df.loc[len(self.metrics_df)] = ["balanced_acc", f"{hx_round(balanced_acc_score_train)}", hx_round(balanced_acc_score_eval), hx_round(balanced_acc_score_test)]
        self.metrics_df.loc[len(self.metrics_df)] = ["precision", f"{hx_round(precision_score_train)}", hx_round(precision_score_eval), hx_round(precision_score_test)]

        # --------------------------------------------------------------------------------------------
        # -- 2: Proceso y transformo el metrics_df para dejarlo en el formato adecuado
        # --------------------------------------------------------------------------------------------

        # ---- 2.1: Eliminao columnas que contengan únicamente valores NaN
        self.metrics_df = self.metrics_df.dropna(axis=1, how="all")

        # ---- 2.2: Convertir a float64 todas las columnas excepto 'metric'
        self.metrics_df.loc[:, self.metrics_df.columns != 'metric'] = self.metrics_df.loc[:, self.metrics_df.columns != 'metric'].astype('float64')

        # --------------------------------------------------------------------------------------------
        # -- 3: Pinto el df y generoel diccionario de métricas
        # --------------------------------------------------------------------------------------------

        # ---- 3.1: Pinto el df
        self.CT.IT.print_tabulate_df(self.metrics_df)

        # ---- 3.2: Creo el diccionario de metricas para graficarlo finalmente en el genethic con todos los individuos
        gen_metric_dict: dict = {
            "acc_train": self.metrics_df[self.metrics_df["metric"] == "accuracy"][f"train_{self.target_col_name}"].iloc[0],
            "acc_test": self.metrics_df[self.metrics_df["metric"] == "accuracy"][f"test_{self.target_col_name}"].iloc[0],
            "recall_train": self.metrics_df[self.metrics_df["metric"] == "recall"][f"train_{self.target_col_name}"].iloc[0],
            "recall_test": self.metrics_df[self.metrics_df["metric"] == "recall"][f"test_{self.target_col_name}"].iloc[0],
            "spec_train": self.metrics_df[self.metrics_df["metric"] == "specificity"][f"train_{self.target_col_name}"].iloc[0],
            "spec_test": self.metrics_df[self.metrics_df["metric"] == "specificity"][f"test_{self.target_col_name}"].iloc[0],
            "f1_train": self.metrics_df[self.metrics_df["metric"] == "f1"][f"train_{self.target_col_name}"].iloc[0],
            "f1_test": self.metrics_df[self.metrics_df["metric"] == "f1"][f"test_{self.target_col_name}"].iloc[0],
            "roc_train": self.metrics_df[self.metrics_df["metric"] == "roc_auc"][f"train_{self.target_col_name}"].iloc[0],
            "roc_test": self.metrics_df[self.metrics_df["metric"] == "roc_auc"][f"test_{self.target_col_name}"].iloc[0],
            "bal_acc_train": self.metrics_df[self.metrics_df["metric"] == "balanced_acc"][f"train_{self.target_col_name}"].iloc[0],
            "bal_acc_test": self.metrics_df[self.metrics_df["metric"] == "balanced_acc"][f"test_{self.target_col_name}"].iloc[0],
            "precision_train": self.metrics_df[self.metrics_df["metric"] == "precision"][f"train_{self.target_col_name}"].iloc[0],
            "precision_test": self.metrics_df[self.metrics_df["metric"] == "precision"][f"test_{self.target_col_name}"].iloc[0],
        }

        del self.metrics_df

        # ---- 3.3: Devuelvo el diccionario con la metrica seleccionada y las metricas totales
        result: Dict[str, Any] = {}

        if self.selected_metric is not None:
            match self.selected_metric:
                case "accuracy":
                    result: Dict[str, Any] = {"metric_train": gen_metric_dict["acc_train"], "metric_eval": accuracy_score_eval, "metric_test": gen_metric_dict["acc_test"],
                            "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "specificity":
                    result: Dict[str, Any] = {"metric_train": gen_metric_dict["spec_train"], "metric_eval": specificity_eval, "metric_test": gen_metric_dict["spec_test"],
                            "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "balanced_acc":
                    result: Dict[str, Any] = {"metric_train": gen_metric_dict["bal_acc_train"], "metric_eval": balanced_acc_score_eval, "metric_test": gen_metric_dict["bal_acc_test"],
                            "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "f1":
                    result: Dict[str, Any] = {"metric_train": gen_metric_dict["f1_train"], "metric_eval": f1_score_eval, "metric_test": gen_metric_dict["f1_test"],
                            "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "precision":
                    result: Dict[str, Any] = {"metric_train": gen_metric_dict["precision_train"], "metric_eval": precision_score_eval, "metric_test": gen_metric_dict["precision_test"],
                            "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "recall":
                    result: Dict[str, Any] = {"metric_train": gen_metric_dict["recall_train"], "metric_eval": recall_score_eval, "metric_test": gen_metric_dict["recall_test"],
                            "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "roc_auc":
                    result: Dict[str, Any] = {"metric_train": gen_metric_dict["roc_train"], "metric_eval": roc_auc_score_eval, "metric_test": gen_metric_dict["roc_test"],
                            "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}

                case _:
                    raise ValueError("La metrica seleccionada no está contemplada")

            # ---- 3.4: Retorno el resultado
            return result

        return None