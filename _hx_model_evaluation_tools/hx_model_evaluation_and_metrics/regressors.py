from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error
from _hx_model_evaluation_tools import HxFeatureImportancesandCorrelations
from _hx_model_evaluation_tools.hx_model_evaluation_graphics import HxModelEvaluationGraphics
from constants_and_tools import ConstantsAndTools
import plotly.graph_objects as go
import plotly.express as px
from typing import List
import pandas as pd
import numpy as np


class RegressorMetricsCalculations:
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

        :param probs_result_df: Diccionario que contiene el y_test real, y el y_predicho
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
        self.bins_dict: dict | None = bins_dict

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
        # -- 2: Ejecutamos evaluate regression para realizar todas las tareas de evaluacion y retornamos
        # ----------------------------------------------------------------------------------------------------

        # ---- 2.1: Llamo al metodo de evaluar regresion y obtengo las metricas
        metrics_dict: dict = self.evaluate_regression()

        # ---- 2.2: Retorno las metricas (no lo hago directo porque me facilita el debug si es necesario)
        return metrics_dict

    def evaluate_regression(self):
        """
        Metodo para obtener un diccionario de métricas de cada columna target
        :return:
        """
        # ----------------------------------------------------------------------------------------------------
        # -- 1: Obtengo las métricas y las almaceno
        # ----------------------------------------------------------------------------------------------------

        # ---- 1.1: Creo el diccionario de metricas
        all_metrics_dict: dict = {}

        # ---- 1.2: MSE (Mean Squared Error)
        mse = mean_squared_error(self.probs_result_df[f'real_value_{self.target_col_name}'], self.probs_result_df[f'predict_value_{self.target_col_name}'])

        # ---- 1.3: MAE (Mean Absolute Error)
        mae = mean_absolute_error(self.probs_result_df[f'real_value_{self.target_col_name}'], self.probs_result_df[f'predict_value_{self.target_col_name}'])

        # ---- 1.4: R2 (Coeficiente de Pearson)
        r2 = r2_score(self.probs_result_df[f'real_value_{self.target_col_name}'], self.probs_result_df[f'predict_value_{self.target_col_name}'])

        # ---- 1.5: RMSE (Root Mean squared error)
        rmse = np.sqrt(mse)

        # ---- 1.6: MSLE (Mean squared log error) and RMSLE (Root Mean squared log error)
        try:
            # ------ 1.6.1: MSLE
            msle = mean_squared_log_error(self.probs_result_df[f'real_value_{self.target_col_name}'], self.probs_result_df[f'predict_value_{self.target_col_name}'])

            # ------ 1.6.2: RMSLE
            rmsle = np.sqrt(msle)

        except ValueError:
            msle = np.nan
            rmsle = np.nan

        # ---- 1.7: MAPE (Mean absolute percentage error)
        mape = mean_absolute_percentage_error(self.probs_result_df[f'real_value_{self.target_col_name}'], self.probs_result_df[f'predict_value_{self.target_col_name}'])

        # ----------------------------------------------------------------------------------------------------
        # -- 2: Pinto las métricas por consola
        # ----------------------------------------------------------------------------------------------------

        self.CT.IT.sub_intro_print(f"RegressorMetricsCalculation 1: Métricas {self.model_name} {self.target_col_name}")

        self.CT.IT.info_print(f"MSE (Mean Squared Error):                {round(mse, 4)}", "light_cyan")
        self.CT.IT.info_print(f"MAE (Mean Absolute Error):               {round(mae, 4)}", "light_cyan")
        self.CT.IT.info_print(f"R2 (Coeficiente de Pearson):             {round(r2, 4)}", "light_cyan")
        self.CT.IT.info_print(f"RMSE (Root Mean squared error):          {round(rmse, 4)}", "light_cyan")
        self.CT.IT.info_print(f"MSLE (Mean squared log error):           {round(msle, 4)}", "light_cyan")
        self.CT.IT.info_print(f"RMSLE (Root Mean squared log error):     {round(rmsle, 4)}", "light_cyan")
        self.CT.IT.info_print(f"MAPE (Mean absolute percentage error):   {round(mape, 4)}", "light_cyan")

        # ----------------------------------------------------------------------------------------------------
        # -- 3: Calculo y muestro la estratificacion si self.bins no es none
        # ----------------------------------------------------------------------------------------------------

        if self.bins_dict is not None:

            # ---- 3.1:  Creamos la columna de estratificacion utilizando pd.cut(), esto depende de lo que le pasamos con parametro en el bins_dict
            self.probs_result_df[f'stratification_{self.target_col_name}'] = pd.cut(self.probs_result_df[f"predict_value_{self.target_col_name}"],
                                                                                    bins=self.bins_dict["bins"],
                                                                                    labels=self.bins_dict["labels"],
                                                                                    right=False)

            # ---- 3.2: Creamos dos columnas, f'stratification_{self.target_col_name}_high' y f'stratification_{self.target_col_name}_low' para separar los bounds
            self.probs_result_df[f'stratification_{self.target_col_name}_high'] =self.probs_result_df[f'stratification_{self.target_col_name}'].str.split('-').str[0].astype(float)
            self.probs_result_df[f'stratification_{self.target_col_name}_low'] =self.probs_result_df[f'stratification_{self.target_col_name}'].str.split('-').str[1].astype(float)

            # ---- 3.3: Una vez tengo las columnas superior e inferior en float, valido si se ha acertado o fallado en el estrato

            self.probs_result_df[f"aciertos_{self.target_col_name}"] = (
                self.probs_result_df[f"real_value_{self.target_col_name}"]
                .between(
                    self.probs_result_df[f'stratification_{self.target_col_name}_high'],  # Limite superior
                    self.probs_result_df[f'stratification_{self.target_col_name}_low'],  # Limite inferior
                    inclusive='both'
                )
            )

            # ---- 4.2: Creo el stratify df que se va a ir agregando al self.stratify df
            stratify_df_base_cols: list = ["confidence", "total_predicho", "aciertos", "fallos"]
            self.stratify_df: pd.DataFrame = pd.DataFrame(columns=[f'{z}_{self.target_col_name}' if z != "confidence" else f"{z}" for z in stratify_df_base_cols])


            # ---- 4.3: Relleno  iterativamente los bins de estratificacion en base al self.bins_dict
            for bn in self.bins_dict["labels"][::-1]:
                if bn in self.probs_result_df[f"stratification_{self.target_col_name}"].unique().tolist():
                    res_shape: int = self.probs_result_df.shape[0]
                    bn_df = self.probs_result_df[self.probs_result_df[f"stratification_{self.target_col_name}"] == bn]
                    bn_df_aciertos = bn_df[bn_df[f"aciertos_{self.target_col_name}"]]
                    bn_df_fallos = bn_df[(~bn_df[f"aciertos_{self.target_col_name}"])]

                    self.stratify_df.loc[len(self.stratify_df)] = {
                        f"confidence": bn,
                        f"total_predicho_{self.target_col_name}": f"{round(bn_df.shape[0] / res_shape * 100, 2)}%  ({bn_df.shape[0]}/{res_shape})" if res_shape != 0 else pd.NA,
                        f"aciertos_{self.target_col_name}": f"{round(bn_df_aciertos.shape[0] / bn_df.shape[0] * 100, 2)}%  ({bn_df_aciertos.shape[0]}/{bn_df.shape[0]})" if bn_df.shape[0] != 0 else pd.NA,
                        f"fallos_{self.target_col_name}": f"{round(bn_df_fallos.shape[0] / bn_df.shape[0] * 100, 2)}%  ({bn_df_fallos.shape[0]}/{bn_df.shape[0]})" if bn_df.shape[0] != 0 else pd.NA,
                    }

        # ----------------------------------------------------------------------------------------------------
        # -- 4: Creo gráficos, pinto y retorno
        # ----------------------------------------------------------------------------------------------------

        # ---- 4.1: Creo el gráfico de estratificacion en barras
        if self.bins_dict is not None:
            self.plot_stratify_barplot(self.probs_result_df, f"{self.model_save_path}/stratify_barplot_{self.target_col_name}.html",
                                       f"stratification_{self.target_col_name}", f"aciertos_{self.target_col_name}")

        # ---- 4.2: Creo el diccionario con todas las metricas que voy a retornar
        all_metrics_dict[f"{self.target_col_name}"] = {
            "metrics": {
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "rmse": rmse,
                "msle": msle,
                "rmsle": rmsle,
                "mape": mape,
            }
        }

        # ---- 4.3: Pinto la entrada y las metricas de estratificacion
        if self.bins_dict is not None:
            self.CT.IT.sub_intro_print(f"RegressorMetricsCalculation: Tabla de estratificacion {self.model_name}")
            self.CT.IT.print_tabulate_df(self.stratify_df, row_print=100)

        # ---- 4.4: Almaceno el gráfico de la estratificacion en formato tabla
        if self.bins_dict is not None:
            self.GT.plot_dataframe_html(self.stratify_df, save_path=f"{self.model_save_path}/stratify_df.html", save=True)

        # ---- 4.5: Creo el gráfico de dispersión y lo almaceno en html
        self.plot_and_save_dotplot()

        # ---- 4.6: Retorno el diccionario completo de metricas
        return all_metrics_dict

    # <editor-fold desc="Metodos secundarios usados en evaluateRegression   --------------------------------------------------------------------------------------------------">

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

    def plot_and_save_dotplot(self):
        """
        Genera un gráfico de dispersión (dotplot) con Plotly para comparar valores reales vs. predichos
        y lo guarda en formato HTML y PNG, mostrando el gráfico en pantalla completa y con un estilo profesional.
        """

        # Crear el scatter plot interactivo
        fig = go.Figure()

        # Añadir puntos de dispersión
        fig.add_trace(go.Scatter(
            x=self.probs_result_df[f'real_value_{self.target_col_name}'],
            y=self.probs_result_df[f'predict_value_{self.target_col_name}'],
            mode='markers',
            name='Predicciones',
            marker=dict(
                size=8,
                opacity=0.8,
                color='royalblue',
                line=dict(width=0.5, color='white')
            )
        ))

        # Línea de referencia diagonal (y = x)
        min_val = min(
            self.probs_result_df[f'real_value_{self.target_col_name}'].min(),
            self.probs_result_df[f'predict_value_{self.target_col_name}'].min()
        )
        max_val = max(
            self.probs_result_df[f'real_value_{self.target_col_name}'].max(),
            self.probs_result_df[f'predict_value_{self.target_col_name}'].max()
        )

        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Referencia y=x',
            line=dict(color='black', dash='dash', width=2)
        ))

        # Configuración de estilo y pantalla completa
        fig.update_layout(
            title=dict(
                text='Dispersión de predicciones vs valores reales',
                x=0.5,  # centrar título
                xanchor='center',
                font=dict(size=22, family="Arial, Bold")
            ),
            xaxis=dict(
                title=f'Real Value: {self.target_col_name}',
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True
            ),
            yaxis=dict(
                title=f'Predicted Value: {self.target_col_name}',
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True
            ),
            template='plotly_white',
            legend=dict(
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center"
            ),
            autosize=True,
            width=None,
            height=None
        )

        # Mostrar en pantalla completa (modo responsive en navegador)
        fig.update_layout(
            margin=dict(l=50, r=50, t=80, b=80),
        )

        # Guardar el gráfico como HTML (interactivo y pantalla completa)
        fig.write_html(f"{self.model_save_path}/scatter_plot.html", full_html=True, include_plotlyjs='cdn')

    # </editor-fold>

class EvaluateRegressor:
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
        self.target_col_name: str = data_dict["target_col_name"]

        # ---- 1.7: Inicializo en None el df que va a contener las métricas
        self.metrics_df = None

    def calculate_and_print_metrics(self):

        # --------------------------------------------------------------------------------------------
        # -- 1: Pinto y calculo las metricas
        # --------------------------------------------------------------------------------------------

        # ---- 1.1: Pinto la entrada
        self.CT.IT.sub_intro_print("EvaluateRegressor: Calculando metricas de entrenamiento...")

        # --------------------------------------------------------------------------------------------
        # -- 2: Calculo las métricas
        # --------------------------------------------------------------------------------------------

        # ---- 2.1: Mean squared error
        mse_train = mean_squared_error(self.y_train, self.y_pred_train)
        mse_eval = mean_squared_error(self.y_eval, self.y_pred_eval) if self.y_eval is not None else None
        mse_test = mean_squared_error(self.y_test, self.y_pred_test)

        # ---- 2.2: Root Mean squared error
        rmse_train = np.sqrt(mse_train)
        rmse_eval = np.sqrt(mse_eval) if self.y_eval is not None else None
        rmse_test = np.sqrt(mse_test)

        # ---- 2.3: Mean absolute error
        mae_train = mean_absolute_error(self.y_train, self.y_pred_train)
        mae_eval = mean_absolute_error(self.y_eval, self.y_pred_eval) if self.y_eval is not None else None
        mae_test = mean_absolute_error(self.y_test, self.y_pred_test)

        # ---- 2.4: Coeficiente de determinacion
        r2_train = r2_score(self.y_train, self.y_pred_train)
        r2_eval = r2_score(self.y_eval, self.y_pred_eval) if self.y_eval is not None else None
        r2_test = r2_score(self.y_test, self.y_pred_test)

        try:
            # ---- 2.5: Mean squared log error
            msle_train = mean_squared_log_error(self.y_train, self.y_pred_train)
            msle_eval = mean_squared_log_error(self.y_eval, self.y_pred_eval) if self.y_eval is not None else None
            msle_test = mean_squared_log_error(self.y_test, self.y_pred_test)

            # ---- 2.6: Root Mean squared log error
            rmsle_train = np.sqrt(msle_train)
            rmsle_eval = np.sqrt(msle_eval) if self.y_eval is not None else None
            rmsle_test = np.sqrt(msle_test)

        except ValueError:
            # -- Mean squared log error
            msle_train = np.nan
            msle_eval = np.nan
            msle_test = np.nan

            # -- Root Mean squared log error
            rmsle_train = np.nan
            rmsle_eval = np.nan
            rmsle_test = np.nan

        # ---- 2.7: Mean absolute percentage error
        mape_train = mean_absolute_percentage_error(self.y_train, self.y_pred_train)
        mape_eval = mean_absolute_percentage_error(self.y_eval, self.y_pred_eval) if self.y_eval is not None else None
        mape_test = mean_absolute_percentage_error(self.y_test, self.y_pred_test)

        # --------------------------------------------------------------------------------------------
        # -- 3: Almaceno las metricas en el df self.metrics df, las pinto y creo el diccionario de metricas
        # --------------------------------------------------------------------------------------------

        # ---- 3.0: Creo una lambda para redondear
        hx_round = lambda x: round(x, 4) if x is not None else pd.NA

        # ---- 3.1: Almaceno metricas
        self.metrics_df: pd.DataFrame = pd.DataFrame(columns=("metric", f"train_{self.target_col_name}", f"eval_{self.target_col_name}", f"test_{self.target_col_name}"))
        self.metrics_df.loc[len(self.metrics_df)] = ["MSE", f"{hx_round(mse_train)}", hx_round(mse_eval), hx_round(mse_test)]
        self.metrics_df.loc[len(self.metrics_df)] = ["RMSE", f"{hx_round(rmse_train)}", hx_round(rmse_eval), hx_round(rmse_test)]
        self.metrics_df.loc[len(self.metrics_df)] = ["MAE", f"{hx_round(mae_train)}", hx_round(mae_eval), hx_round(mae_test)]
        self.metrics_df.loc[len(self.metrics_df)] = ["R2", f"{hx_round(r2_train)}", hx_round(r2_eval), hx_round(r2_test)]
        self.metrics_df.loc[len(self.metrics_df)] = ["MSLE", f"{hx_round(msle_train)}", hx_round(msle_eval), hx_round(msle_test)]
        self.metrics_df.loc[len(self.metrics_df)] = ["RMSLE", f"{hx_round(rmsle_train)}", hx_round(rmsle_eval), hx_round(rmsle_test)]
        self.metrics_df.loc[len(self.metrics_df)] = ["MAPE", f"{hx_round(mape_train)}", hx_round(mape_eval), hx_round(mape_test)]

        # ---- 3.2: Elimino columnas vacías (para no mostrar el eval con NA si no hay eval) Pinto el df
        self.metrics_df = self.metrics_df.dropna(axis=1, how="all")

        self.CT.IT.print_tabulate_df(self.metrics_df)

        # ---- 3.3: Creo el diccionario de metricas para graficarlo finalmente en el genethic con todos los individuos
        gen_metric_dict: dict = {
            "mse_train": round(mse_train, 4), "mse_test": round(mse_test, 4),
            "rmse_train": round(rmse_train, 4), "rmse_test": round(rmse_test, 4),
            "mae_train": round(mae_train, 4), "mae_test": round(mae_test, 4),
            "r2_train": round(r2_train, 4), "r2_test": round(r2_test, 4),
            "msle_train": round(msle_train, 4), "msle_test": round(msle_test, 4),
            "rmsle_train": round(rmsle_train, 4), "rmsle_test": round(rmsle_test, 4),
            "mape_train": round(mape_train, 4), "mape_test": round(mape_test, 4),
        }

        del self.metrics_df

        # --------------------------------------------------------------------------------------------
        # -- 4: Hago match y devuelvo el diccionario completo de metricas
        # --------------------------------------------------------------------------------------------

        if self.selected_metric is not None:
            match self.selected_metric:
                case "mse":
                    return {"metric_train": mse_train, "metric_eval": mse_eval, "metric_test": mse_test, "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "rmse":
                    return {"metric_train": rmse_train, "metric_eval": rmse_eval, "metric_test": rmse_test, "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "mae":
                    return {"metric_train": mae_train, "metric_eval": mae_eval, "metric_test": mae_test, "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "msle":
                    return {"metric_train": msle_train, "metric_eval": msle_eval, "metric_test": msle_test, "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "rmsle":
                    return {"metric_train": rmsle_train, "metric_eval": rmsle_eval, "metric_test": rmsle_test, "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "mape":
                    return {"metric_train": mape_train, "metric_eval": mape_eval, "metric_test": mape_test, "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case _:
                    raise ValueError("La metrica seleccionada no está contemplada")

        return None