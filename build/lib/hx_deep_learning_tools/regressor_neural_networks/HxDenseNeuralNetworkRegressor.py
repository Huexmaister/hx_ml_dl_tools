from hx_deep_learning_tools.dl_base_model import HxDeepLearningBaseModel
from _hx_model_evaluation_tools import EvaluateRegressor, RegressorMetricsCalculations
import tensorflow as tf
from _hx_model_evaluation_tools import DlShapToolsBinaryRegressor
from typing import Dict, Literal, List, Any, Tuple
import pandas as pd
import numpy as np
import datetime
import os
import json


class HxDenseNeuralNetworkRegressor(HxDeepLearningBaseModel):

    def __init__(self,
                 data_dict: Dict[str, pd.DataFrame],
                 hiperparams: dict,
                 metric: Literal['mse', 'rmse', 'mae', 'msle', 'rmsle', 'mape'],
                 problem_type: Literal['regression'] = 'regression',
                 geometry: Literal[0, 1, 2, 3] = 1,
                 hidden_layers: int = 2,
                 activation: Literal['relu', 'tanh', 'sigmoid'] = 'relu',
                 regularization_type: Literal['l1', 'l2'] = 'l2',
                 optimizer: Literal['sgd', 'adam', 'rmsprop'] = 'adam',
                 loss_function: Literal['mse', 'mae', 'msle', 'mape', 'huber', 'log_cosh'] = 'mse',
                 include_dropout: bool = True,
                 include_batch_norm: bool = True,
                 early_stopping_patience: int = 15,
                 save_path: str | None = None,
                 bins: dict | None = None,
                 verbose: Literal['show_all', 'show_basics', 'dont_show'] = 'dont_show',
                 random_state: int = 42):
        """
        Clase para construir modelos de clasificación binaria con Redes Neuronales Densas
        :param data_dict: Diccionario que contiene el x_train, y_train, x_test, y_test
        :param save_path: Path donde guardar los resultados
        :param hiperparams: Hiperparámetros del modelo
        :param bins: Bins para discretización
        """
        super().__init__(data_dict,
                         hiperparams,
                         "DNN_Regressor",
                         geometry,
                         hidden_layers,
                         activation,
                         regularization_type,
                         optimizer,
                         include_dropout,
                         include_batch_norm,
                         early_stopping_patience,
                         save_path)

        # --------------------------------------------------------------------------------------------
        # -- 0: Almaceno nombre de la clase, pinto la entrada y creo la propiedad model
        # --------------------------------------------------------------------------------------------

        # ---- 0.1: Almaceno el nombre de la clase
        self.class_name: str = f"HxDenseNeuralNetworkRegressor"

        # ---- 0.2: Pinto la entrada
        self.CT.IT.intro_print(self.class_name)

        # ---- 0.3: Creo la propiedad model y la inicializo en None
        self.model = None

        # ---- 0.4: Establecer semilla para reproducibilidad
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        # --------------------------------------------------------------------------------------------
        # -- 1: Asigno las propiedades abstractas
        # --------------------------------------------------------------------------------------------

        # ---- 1.1: Verbose
        self._verbose: Literal['show_all', 'show_basics', 'dont_show'] = verbose

        # ---- 1.2: Metric
        self._metric: Literal['mse', 'rmse', 'mae', 'msle', 'rmsle', 'mape'] = metric

        # ---- 1.3: Problem type
        self._problem_type: Literal['regression'] = problem_type

        # --------------------------------------------------------------------------------------------
        # -- 2: Asigno las propiedades estandar
        # --------------------------------------------------------------------------------------------

        # ---- 2.1: Establezco los bins
        self.bins = bins

        # ---- 2.2: Path donde se va a guardar la informacion del modelo
        self.model_save_path: str = self.get_save_path()

        # ---- 2.3: Random state con el que se van a inicializar los pesos, por defecto 42
        self.random_state: int = random_state

        # ---- 2.4: Creo los callbacks llamando al metodo de la clase padre
        self._setup_training_callbacks()

        # --------------------------------------------------------------------------------------------
        # -- 3: Gestiono las funciones de pérdida
        # --------------------------------------------------------------------------------------------

        # ---- 3.1: Almaceno en propiedad la funcion de perdida
        self.loss_function: Literal['mse', 'mae', 'msle', 'mape', 'huber', 'log_cosh'] = loss_function

        # ---- 3.2: Defino el dicionario de funciones de perdida
        self.loss_functions_map_dict = {
            'mse': 'mse',  # Mean Squared Error
            'rmse': 'mse',  # RMSE no es diferenciable → usar MSE como proxy
            'mae': 'mae',  # Mean Absolute Error
            'msle': 'msle',  # Mean Squared Logarithmic Error
            'rmsle': 'msle',  # RMSLE no es diferenciable → usar MSLE como proxy
            'mape': 'mape',  # Mean Absolute Percentage Error

            # Funciones de pérdida avanzadas (pueden ser usadas directamente)
            'huber': 'huber',  # Combina MSE y MAE
            'log_cosh': 'log_cosh',  # Log Cosh Loss
            'default': 'mse'  # Por defecto (estándar)
        }

        # ---- 3.3: Defino la lista de funciones de pérdida permitidas
        self.available_loss_functions: List[str] = ['mse', 'mae', 'msle', 'mape', 'huber', 'log_cosh']

        # ---- 3.4: machaco la loss function por si acaso
        self.loss_function: Literal['mse', 'mae', 'msle', 'mape', 'huber', 'log_cosh'] = self.get_loss_function()

    # <editor-fold desc="Metodo privado para crear la red    -------------------------------------------------------------------------------------------------">

    def _build_dnn(self, input_shape: tuple) -> tf.keras.Model:
        """
        Construye una DNN avanzada para clasificación binaria con configuración flexible de capas
        :param input_shape: Forma de los datos de entrada
        :return: Modelo de Keras compilado
        """

        # --------------------------------------------------------------------------------------------
        # -- 1: Obtengo los hipermarámetros y la arquitectura de la red
        # --------------------------------------------------------------------------------------------

        # ---- 1.1: Hiperparámetros básicos (vienen de hiperparams y deben ser numericos)
        units = self.hiperparams.get('units', 64)
        dropout_rate = self.hiperparams.get('dropout', 0.3)
        learning_rate = self.hiperparams.get('learning_rate', 0.001)
        regularization = self.hiperparams.get('regularization', 0.01)

        # ---- 1.2: Hiperparámetros de configuracion (se pasan como parametro individualmente)
        hidden_layers = self.hidden_layers
        activation = self.activation
        geometry = self.geometry_symbol  # '<>', '><', '>', '<'
        include_dropout = self.include_dropout
        include_batch_norm = self.include_batch_norm
        optimizer_type = self.optimizer

        # ---- 1.3: Regularizador (L1 o L2)
        if self.regularization_type == 'l1':
            regularizer = tf.keras.regularizers.L1(regularization)
        else:
            regularizer = tf.keras.regularizers.L2(regularization)

        # --------------------------------------------------------------------------------------------
        # -- 2: Pinto arquitectura y creo el objeto model
        # --------------------------------------------------------------------------------------------

        # ---- 2.1: Pinto la arquitectura de la red
        self.CT.IT.sub_intro_print("NetworkSpecifications")
        self.CT.IT.info_print(f"Arquitectura de red: {geometry}")
        self.CT.IT.info_print(f"Capas: entrada + {hidden_layers} ocultas + salida")
        self.CT.IT.info_print(f"Unidades base: {units}, Dropout: {dropout_rate}, Activación: {activation}")

        # ---- 2.2: Calculao el punto de inflexión para geometrías simétricas
        inflection_point = hidden_layers // 2

        # ---- 2.3: Defino el modelo vacío
        model = tf.keras.Sequential(name="advanced_dnn_regressor")

        # --------------------------------------------------------------------------------------------
        # -- 3: Realizo la configuracion dinamica de la red
        # --------------------------------------------------------------------------------------------

        # ---- 3.1: Agrego capa de activación
        model.add(tf.keras.layers.Dense(
            units=units,
            activation=activation,
            input_shape=input_shape,
            kernel_regularizer=regularizer,
            name="input_dense"
        ))

        # ---- 3.2: Incluyo capa de normalización si así está definido
        if include_batch_norm:
            model.add(tf.keras.layers.BatchNormalization(name="input_bn"))

        # ---- 3.2: Incluyo capa de dropout si así está definido
        if include_dropout:
            model.add(tf.keras.layers.Dropout(dropout_rate, name="input_dropout"))

        # ---- 3.3: Itero para realizar la configuración de las hidden layers
        for i in range(1, hidden_layers + 1):

            # ------ 3.3.1: Defino caracteristicas para evitar el error de reference
            coef: int = 1
            layer_type = "bottleneck"

            # ------ 3.3.2: Hago el matchpara obtener el coeficiente y el tipo de capa
            match geometry:
                case "<>":  # Decrece y luego crece (hourglass)
                    if i <= inflection_point:
                        coef = 2 ** i
                    else:
                        coef = 2 ** ((hidden_layers + 1) - i)
                    layer_type = "bottleneck"

                case "><":  # Crece y luego decrece (diamond)
                    if i <= inflection_point:
                        coef = 2 ** ((hidden_layers + 1) - i)
                    else:
                        coef = 2 ** i
                    layer_type = "expansion"

                case ">":  # Solo decrece (funnel)
                    coef = 2 ** ((hidden_layers + 1) - i)
                    layer_type = "funnel"

                case "<":  # Solo crece (inverse funnel)
                    coef = 2 ** i
                    layer_type = "inverse_funnel"

                case _:  # Constante (fallback)
                    coef = 1
                    layer_type = "constant"

            # ------ 3.3.3: Calculo las unidades para esta capa (asigno un minimo de 8)
            current_units = max(int(units * coef), 8)

            # ------ 3.3.4: Agrego la capa al modelo
            model.add(tf.keras.layers.Dense(
                units=current_units,
                activation=activation,
                kernel_regularizer=regularizer,
                name=f"hidden_{layer_type}_{i}_dense"
            ))

            # ------ 3.3.5: Agrego batch normalization si esta dispuesto así
            if include_batch_norm:
                model.add(tf.keras.layers.BatchNormalization(name=f"hidden_bn_{i}"))

            # ------ 3.3.6: Agrego dropout si esta dispuesto así
            if include_dropout:
                model.add(tf.keras.layers.Dropout(dropout_rate, name=f"hidden_dropout_{i}"))

            # ------ 3.3.7: Pinto la arquitectura de la capa
            self.CT.IT.info_print(f"Capa {i}: {current_units} unidades (coef: {coef})", "cyan")

        # ---- 3.4: Agrego la capa de salida
        model.add(tf.keras.layers.Dense(
            units=1,  # 1 neurona para una sola target
            activation='linear',
            name="output_dense"
        ))

        # --------------------------------------------------------------------------------------------
        # -- 4: Pinto el resumen de la arquitectura, selecciono optimizador y compilo
        # --------------------------------------------------------------------------------------------

        # ---- 4.1: Obtengo los parametros y pinto
        total_params = model.count_params()
        self.CT.IT.info_print(f"Arquitectura finalizada: {geometry} con {hidden_layers} capas ocultas")
        self.CT.IT.info_print(f"Total de parámetros: {total_params:,}")

        # ---- 4.2: Selecciono el optimizador
        if optimizer_type.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                amsgrad=False
            )
        elif optimizer_type.lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate,
                rho=0.9,
                momentum=0.0,
                epsilon=1e-7
            )
        else:
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=0.9,
                nesterov=True
            )

        # ---- 4.3: Agrego las métricas (solo se ven si hay verbose)
        metrics = [
            'mae',
            'mse',
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
            tf.keras.metrics.R2Score(name='r2')
        ]

        # ---- 4.4: Compilo el modelo
        model.compile(
            optimizer=optimizer,
            loss=self.loss_function,
            metrics=metrics
        )

        # ---- 4.5: Retorno el modelo
        return model

    # </editor-fold>

    # <editor-fold desc="Implementacion de metodos abstractos">

    def get_save_path(self) -> str:
        """
        Metodo que crea la estructura de carpetas para guardar el modelo
        """
        if not os.path.exists(f"{self.save_path}/{self.current_train_version}/{self.model_name}"):
            self.CT.OT.create_folder_if_not_exists(f"{self.save_path}/{self.current_train_version}/{self.model_name}")
        else:
            self.current_train_version: str = self.new_training_version()
            self.CT.OT.create_folder_if_not_exists(f"{self.save_path}/{self.current_train_version}")
            self.CT.OT.create_folder_if_not_exists(f"{self.save_path}/{self.current_train_version}/{self.model_name}")

        self.day_train_versions: List[str] = self.list_versions()
        return f"{self.save_path}/{self.current_train_version}/{self.model_name}"

    def fit_and_get_metrics(self, save_metrics_dict: bool = True) -> Dict[str, Any]:
        """
        Entrena la red neuronal y devuelve las metricas
        :param save_metrics_dict:
        :return:
        """

        # --------------------------------------------------------------------------------------------
        # -- 0: Pinto la entrada
        # --------------------------------------------------------------------------------------------

        start_date: datetime.datetime = datetime.datetime.now()
        self.CT.IT.sub_intro_print(f"{self.class_name}.fit_and_get_metrics: Inicio de entrenamiento a las {start_date}...")

        # --------------------------------------------------------------------------------------------
        # -- 1: Construyo el modelo y entreno
        # --------------------------------------------------------------------------------------------

        try:
            # ---- 1.1: Obtengo el batch sample, el feature data y el inpit_shape del tensor
            sample_batch = next(iter(self.train_dataset.take(1)))
            feature_data = sample_batch[0]
            input_shape = (feature_data.shape[-1],)

            # ---- 1.2: Buildeo el modelo
            self.model = self._build_dnn(input_shape)

            # ---- 1.3: Pinto la arquitectura del mddelo si el verbose está en True
            if self.verbose:
                print(self.model.summary())
                self.CT.IT.info_print(f"Modelo creado con {self.model.count_params():,} parámetros")

            # ---- 1.4: Realizo el entrenamiento
            self.model.fit(
                self.train_dataset,
                epochs=self.hiperparams.get('epochs', 100),
                validation_data=self.test_dataset,
                callbacks=self.callbacks,
                verbose=self.verbose,
                shuffle=False
            )

            # --------------------------------------------------------------------------------------------
            # -- 2: Realizo las predicciones y formateo los resultados
            # --------------------------------------------------------------------------------------------

            # ---- 2.1: Realizo las predicciones
            y_pred_train = self.model.predict(self.x_train_df.values, verbose=0)
            y_pred_test = self.model.predict(self.x_test_df.values, verbose=0)

            # ---- 2.2. Obtengo solo las probabilidades de la clase positiva
            # y_pred_train = y_pred_train.flatten()
            # y_pred_test = y_pred_test.flatten()

            # ---- 2.3. Convierto probabilidades a clases (threshold 0.5)
            # y_pred_train_classes = (y_pred_train > 0.5).astype(int)
            # y_pred_test_classes = (y_pred_test > 0.5).astype(int)

            # --------------------------------------------------------------------------------------------
            # -- 3: Obtengo métricas y retorno
            # --------------------------------------------------------------------------------------------

            # ---- 3.1: Ejecuto la evaluación
            metrics: Dict[str, Any] = EvaluateRegressor({"y_train": self.y_train,
                                                         "y_eval": None,
                                                         "y_test": self.y_test,
                                                         "y_pred_train": y_pred_train,
                                                         "y_pred_eval": None,
                                                         "y_pred_test": y_pred_test,
                                                         "target_col_name": self.target_col_name},
                                                        self.model_name,
                                                        self.metric).calculate_and_print_metrics()

            # ---- 3.2: Guardo el dict de metricas si asi está especificado
            if save_metrics_dict:
                self.save_metrics_dict_json(metrics, self.model_save_path)

            # ---- 3.3: Pinto la salida
            end_date = datetime.datetime.now()
            self.CT.IT.info_print(f"fit_and_get_metrics terminado a las {end_date}. Duración: {end_date - start_date}", "light_magenta")

            # ---- 3.4: Retorno
            return metrics

        except Exception as e:
            self.CT.IT.info_print(f"Error durante el entrenamiento: {e}")
            raise

    def fit_and_get_model_and_results(self) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any], List]:
        """
        Entrena y devuelve el modelo completo con resultados
        :return: Tupla con (modelo, probabilidades test, probabilidades train, métricas, pesos)
        """
        # --------------------------------------------------------------------------------------------
        # -- 1: Pinto entrada y utilizo el metodo self.fit_and_get_metrics para entrenar el modelo
        # --------------------------------------------------------------------------------------------

        # ---- 1.1: Pinto la entrada para tener referencia de lo que tarda
        start_date: datetime.datetime = datetime.datetime.now()
        self.CT.IT.sub_intro_print(f"{self.class_name}.fit_and_get_model_and_results: Inicio de entrenamiento a las {start_date}...")

        # ---- 1.2: Entreno el modelo y evalúo metricas usando self.fit_and_get_metrics(save_metrics_dict=False)
        self.fit_and_get_metrics(save_metrics_dict=False)

        # --------------------------------------------------------------------------------------------
        # -- 2: Realizo las predicciones y obtengo las probabilidades positivas de train y test
        # --------------------------------------------------------------------------------------------

        # ---- 2.1: Realizo predicciones sobre el conjunto de train y test
        x_train_probs = self.model.predict(self.x_train_df.values, verbose=0)
        x_test_probs = self.model.predict(self.x_test_df.values, verbose=0)

        # ---- 2.2: Me quedo con arrays (n, ) que contienen las probabilidades positivas
        # x_train_probs: np.ndarray = x_train_probs[:, 0]
        # x_test_probs: np.ndarray = x_test_probs[:, 0]
        x_train_probs = x_train_probs.flatten()
        x_test_probs = x_test_probs.flatten()

        # --------------------------------------------------------------------------------------------
        # -- 3: Obtengo los pesos de la primera capa, proceso datos y evalúo el clasificador
        # --------------------------------------------------------------------------------------------

        # ---- 3.1: Obtengo los pesos (Codigo de GPT, validar cruzando con los pesos de ML)
        model_weights, model_weights_dict = self.get_weights(self.model)

        # ---- 3.2: Creo el dict que contiene los y_realies y los y_predichos de test
        probs_result_dict = {
            f'real_value_{self.target_col_name}': self.y_test.values.ravel(),
            f'predict_value_{self.target_col_name}': x_test_probs
        }

        # ---- 3.3: Transformo a dataframe para mas comodidad
        probs_result_df = pd.DataFrame(probs_result_dict)

        # ---- 3.4: Evalúo las métricas y obtengo el diccionario
        metrics_dict = RegressorMetricsCalculations(
            probs_result_df,
            self.model_save_path,
            self.x_test_df,
            self.model,
            model_weights,
            self.target_col_name,
            self.model_name,
            self.bins
        ).run()

        # --------------------------------------------------------------------------------------------
        # -- 4: Construyo el diccionario de salida y retorno la tupla
        # --------------------------------------------------------------------------------------------

        # ---- 4.1: Agrego la informacion al master_result_dict
        self.master_result_dict[f"{self.model_name}"] = {
            "columns": list(self.x_train_df.columns),
            "target": self.target_col_name,
            "best_params": self.hiperparams,
            "geometry": self.geometry,
            "geometry_symbol": self.geometry_symbol,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "regularization_type": self.regularization_type,
            "optimizer": self.optimizer,
            "include_dropout": self.include_dropout,
            "include_batch_norm": self.include_batch_norm,
            "early_stopping_patience": self.early_stopping_patience,
            "metrics": metrics_dict,
            "model_weights": model_weights_dict,
            "model_class_name": self.model.__class__.__name__,
            "model_module_name": self.model.__class__.__module__,
            "model_architecture": json.loads(self.model.to_json()),
            "model_summary": self.get_model_summary(self.model)
        }




        # ---- 4.2: Guardo el modelo en formato .keras (¡IMPORTANTE! Desde tf 2.12 se usa .keras, para tf_gpu 2.10 sigue siendo .h5)
        self.model.save(os.path.join(self.model_save_path, 'dnn_regressor.keras'))

        # ---- 4.3: Almaceno el self.master_dict
        self.save_metrics_dict_json(self.master_result_dict, self.model_save_path, "complete_train_result")

        # ---- 4.4: Pinto salida y retorno la tupla
        end_date = datetime.datetime.now()
        self.CT.IT.info_print(f"fit_and_get_model_and_results terminado a las {end_date}. Duración: {end_date - start_date}", "light_magenta")

        return self.model, x_test_probs, x_train_probs, self.master_result_dict, model_weights

    def get_model_save_path(self) -> str:
        return self.model_save_path

    def execute_shap_analysis(self, sample: bool = True, num_features_to_show: int = 100, num_sample: int = 200, background_sample: int = 100) -> pd.DataFrame:
        """
        Metodo para ejecutar el analisis SHAP de un modelo de Deep Learning
        :param sample:
        :param num_features_to_show:
        :param num_sample:
        :param background_sample:
        :return:
        """
        return DlShapToolsBinaryRegressor(self.x_test_df, self.model_name, self.model_save_path, self.model, sample, num_features_to_show, num_sample, background_sample).run()

    def get_loss_function(self) -> Literal['mse', 'mae', 'msle', 'mape', 'huber', 'log_cosh']:
        """
        Metodo para obtener el loss function dentro del rango permitido
        :return:
        """

        # -- 1: Valido si la funcion de perdida está permitida, si es así, adelante, sino, se avisa de que se pone el default
        if self.loss_function in self.available_loss_functions:
            self.CT.IT.info_print(f"La función de pérdida (loss function) es: {self.loss_function}", "light_magenta")
        else:
            self.CT.IT.info_print(f"La función de pérdida (loss function) {self.loss_function} no está permitida, ponemos default 'mse'", "light_yellow")

        # -- 2: Retorno la funcion de perdida
        # noinspection PyTypeChecker
        return self.loss_functions_map_dict.get(self.metric.lower(), self.loss_functions_map_dict['default'])

    # </editor-fold>

    # <editor-fold desc="Implementacion de propiedades abstractas">

    @property
    def verbose(self):
        verbose_dict = {'show_all': 1, 'show_basics': 1, 'dont_show': 0}
        return verbose_dict[self._verbose]

    @property
    def metric(self):
        if self._metric not in ['mse', 'rmse', 'mae', 'msle', 'rmsle', 'mape']:
            raise ValueError(f'Invalid metric: {self._metric}')
        return self._metric

    @property
    def problem_type(self):
        if self._problem_type not in ['regression']:
            raise ValueError(f'Invalid problem_type: {self._problem_type}. Must be regression')
        return self._problem_type

    # </editor-fold>