from hx_ml_dl_tools import HxLightGbmRegressor, HxXtremeGradientBoostingRegressor, HxCatBoostRegressor
from hx_ml_dl_tools import HxDenseNeuralNetworkRegressor
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hx_hyperparameters_builder import *
from info_tools import InfoTools
import pandas as pd


class DiabetesExample:
    def __init__(self):
        """
        Ejemplo de regresion de modelos de ML y DL para el ejemplo de la diabetes de sklearn
        """

        # --------------------------------------------------------------------------------------
        # -- 0: Instancio InfoTools y pinto la entrada
        # --------------------------------------------------------------------------------------

        # ---- 0.1: Instancia de InfoTools
        self.IT: InfoTools = InfoTools()

        # ---- 0.2: Pinto la entrada
        self.IT.header_print(f"DiabetesExample: Ejemplo de regresion con sklearn.datasets.load_diabetes utilizando hx_ml_dl_tools")

        # --------------------------------------------------------------------------------------
        # -- 1: Descargo el dataset y obtengo el data dict
        # --------------------------------------------------------------------------------------

        # ---- 1.1: Descargo el dataset
        self.diabetes_dataset = load_diabetes()

        # ---- 1.2: Transformo a dataframe (Lo requieren los modelos)
        self.x = pd.DataFrame(self.diabetes_dataset.data, columns=self.diabetes_dataset.feature_names)
        self.y = pd.DataFrame(self.diabetes_dataset.target, columns=["target"])

        # ---- 1.3: Divido en train y test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        # ---- 1.4: Creo el diccionario en el formato requerido por los modelos
        self.data_dict = {"x_train": self.x_train, "y_train": self.y_train, "x_test": self.x_test, "y_test": self.y_test}

        # --------------------------------------------------------------------------------------
        # -- 2: Creo a mano el diccionario de bins para estratificar
        # --------------------------------------------------------------------------------------

        # ---- 2.1: Este diccionario (OPCIONAL) sirve para categorizar las predicciones del regresor y poder observar donde acierta y falla mas
        self.bins: dict = {
            "bins": [0, 50, 100, 150, 200, 250, 300, 350, 400],
            "labels": ["0-50", "50-100", "100-150", "150-200", "200-250", "250-300", "300-350", "350-400"]
        }

        # ---- 2.2: Información del dataset
        self.IT.sub_intro_print(f"Dataset Diabetes cargado:")
        self.IT.info_print(f"Muestras totales: {self.x.shape[0]}")
        self.IT.info_print(f"Características: {self.x.shape[1]}")
        self.IT.info_print(f"Rango target: {self.y['target'].min():.2f} - {self.y['target'].max():.2f}")
        self.IT.info_print(f"Train samples: {self.x_train.shape[0]}")
        self.IT.info_print(f"Test samples: {self.x_test.shape[0]}")

        # ---- 2.3: Informacion de distribucion
        self.get_dataset_info()

    def execute_example(self):
        """
        Metodo que ejecuta en cadena los metodos de entrenamiento
        :return:
        """

        self._light_gbm_regressor()
        self._xg_boost_regressor()
        self._cat_boost_regressor()
        self._dnn_regressor()

    def get_dataset_info(self):
        """
        Metodo adicional para obtener información detallada del dataset
        """
        self.IT.sub_intro_print("Estadísticas del target:")

        self.IT.info_print(f"Media: {self.y['target'].mean():.2f}")
        self.IT.info_print(f"Mediana: {self.y['target'].median():.2f}")
        self.IT.info_print(f"Std: {self.y['target'].std():.2f}")
        self.IT.info_print(f"Min: {self.y['target'].min():.2f}")
        self.IT.info_print(f"Max: {self.y['target'].max():.2f}")

    # <editor-fold desc="Modelos   -----------------------------------------------------------------------------------------------------------------------------------------------">

    def _dnn_regressor(self):
        """
        Metodo para realizar el entrenamiento de la red neuronal densa de regresion
        :return:
        """

        # --------------------------------------------------------------------------------------
        # -- 1: Obtengo los hiperparametros que le voy a pasar
        # --------------------------------------------------------------------------------------

        # ---- 1.1: Opcion de pasarselos a mano
        # dnn_hyperparams: Dict[str, Union[int, float]] = {'units': 8, 'dropout': 0.5, 'learning_rate': 0.01, 'regularization': 0.1, 'epochs': 200}

        # ---- 1.2: Opcion de usar la clase HxDnnHyperparameterBuilder

        # ------ 1.2.1: Obtener parámetros aleatorios dentro del rango preestablecido en HxDnnHyperparameterBuilder
        # dnn_hyperparams: Dict[str, Union[int, float]] = HxDnnHyperparameterBuilder().get_random_hyperparams()

        # ------ 1.2.2: Pasarle los hiperparametros al constructor para que valide si hay alguno que se usa poco o no existe (USAMOS ESTA PARA EL EJEMPLO)
        dnn_hyperparams = HxDnnHyperparameterBuilder({'units': 8, 'dropout': 0.5, 'learning_rate': 0.01, 'regularization': 0.1, 'epochs': 200}).validate_and_get_hyperparams()

        # --------------------------------------------------------------------------------------
        # -- 2: Instancio la clase que me va a proporcionar un toolkit para ese modelo en concreto
        # --------------------------------------------------------------------------------------

        # ---- 2.1: Le paso los hiperparametros deseados (Entrar en la clase para ver detalle)
        dnn: HxDenseNeuralNetworkRegressor = HxDenseNeuralNetworkRegressor(
            self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            dnn_hyperparams,  # Diccionario de hiperparametros
            'mae',  # Literal['mse', 'rmse', 'mae', 'msle', 'rmsle', 'mape'] (por defecto 'mse')
            'regression',  # Literal['regression'] = 'regression' (en este caso, solo puede ser regression)
            1,  # Literal[0, 1, 2, 3] = 1 ({0: "<", 1: ">", 2: "<>", 3: "><"}) Arquitectura de la red
            1,  # Capas ocultas de la red (Toda red tiene 1 capa de entrada + N capas ocultas + 1 capa de salida)
            "relu",  # Literal['relu', 'tanh', 'sigmoid'] = 'relu' (Funcion de activacion)
            "l1",  # Literal['l1', 'l2'] = 'l2'  (Tipo de regularizacion que se aplicará en caso de aplicar regularizacion)
            "adam",  # Literal['sgd', 'adam', 'rmsprop'] = 'adam'  (Optimizador que se va a utilizar)
            "mse",  # Literal['mse', 'mae', 'msle', 'mape', 'huber', 'log_cosh'] = 'mse' (Función de pérdida para el entrenamiento)
            True,  # bool (Incluir o no capas de dropout (en caso de no incluirse, el hiperparametro dropout no hace nada))
            True,  # bool (Incluir capa de batch normalization)
            15,  # Si en N epochs no ha mejorado la métrica, aplica un earlyStop callback y devuelve el mejor modelo hasta la fecha
            None,  # Path donde se almacenarán los resultados (por defecto None, ya que se crean internamente las estructuras de carpetas)
            self.bins,  #  dict | None = None (Diccionario de estratificacion)
            "dont_show",  # Literal['show_all', 'show_basics', 'dont_show'] = 'dont_show' (Cantidad de informacion adicional que se muestra en consola)
            42,  # Random state de incializacion de pesos para garantizar reproducibilidad
        )

        # ---- 2.2: [Metodo]: Permite entrenar el modelo, pintar las metricas en pantalla y devolver un diccionario de las metricas
        # dnn.fit_and_get_metrics(save_metrics_dict=False)

        # ---- 2.3: [Metodo]: Metodo principal de la clase que ejecuta internamente 'fit_and_get_metrics', almacena el modelo junto con graficos e info adicional
        dnn.fit_and_get_model_and_results()

        # ---- 2.4: [Metodo]: Permite ejecutar el analisis SHAP del modelo
        dnn.execute_shap_analysis(
            sample=True,  # Realizar un sampleo de la data antes de calcular SHAP (!! Es muy intensivo en cómputo si no se samplea !!)
            num_features_to_show=100,  # En los gráficos se van a mostrar las N características mas importantes (con mas peso)
            num_sample=200,  # Cantidad de muestra (filas) con las que se va a realizar el analisis
            background_sample=100,  # Número de instancias para background (explicador deep/grad).
        )

    def _light_gbm_regressor(self):
        """
        Metodo para realizar el entrenamiento del lgbm de regresion
        :return:
        """

        # --------------------------------------------------------------------------------------
        # -- 1: Obtengo los hiperparametros que le voy a pasar
        # --------------------------------------------------------------------------------------

        # ---- 1.1: En este caso, los obtengo aleatorios (en el ejemplo de la dnn están todas las opciones)
        lgbm_hyperparams: Dict[str, Union[int, float]] = HxLightgbmHyperparameterBuilder().get_random_hyperparams()

        # --------------------------------------------------------------------------------------
        # -- 2: Instancio la clase que me va a proporcionar un toolkit para ese modelo en concreto
        # --------------------------------------------------------------------------------------

        # ---- 2.1: Le paso los hiperparametros deseados (Entrar en la clase para ver detalle)
        lgbm: HxLightGbmRegressor = HxLightGbmRegressor(
            data_dict=self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            hiperparams=lgbm_hyperparams,  # Diccionario con los hiperparametros del LGBM
            metric="mae",  # Literal['mse', 'rmse', 'mae', 'msle', 'rmsle', 'mape'], Metrica que vamos a evaluar
            problem_type="regression_l1",  # Literal['regression', 'regression_l1', 'quantile', 'tweedie'] = 'regression_l1' (funcion objetivo del lgbm)
            save_path=None,  # Path donde se almacenarán los resultados (por defecto None, ya que se crean internamente las estructuras de carpetas)
            bins=self.bins,  # dict | None = None (Diccionario de estratificacion)
            verbose="dont_show",  # Literal['show_all', 'show_basics', 'dont_show'] = 'dont_show' (Cantidad de informacion adicional que se muestra en consola)
            random_state=42,  # Random state de incializacion de pesos para garantizar reproducibilidad

        )

        # ---- 2.2: [Metodo]: Permite entrenar el modelo, pintar las metricas en pantalla y devolver un diccionario de las metricas
        # lgbm.fit_and_get_metrics(save_metrics_dict=False)

        # ---- 2.3: [Metodo]: Metodo principal de la clase que ejecuta internamente 'fit_and_get_metrics', almacena el modelo junto con graficos e info adicional
        lgbm.fit_and_get_model_and_results()

        # ---- 2.4: [Metodo]: Permite ejecutar el analisis SHAP del modelo
        lgbm.execute_shap_analysis(
            sample=True,  # Realizar un sampleo de la data antes de calcular SHAP (!! Es muy intensivo en cómputo si no se samplea !!)
            num_features_to_show=100,  # En los gráficos se van a mostrar las N características mas importantes (con mas peso)
            num_sample=200,  # Número de instancias para background (explicador deep/grad).
        )

    def _xg_boost_regressor(self):
        """
        Metodo para realizar el entrenamiento del xgboost de regresion
        :return:
        """

        # --------------------------------------------------------------------------------------
        # -- 1: Obtengo los hiperparametros que le voy a pasar
        # --------------------------------------------------------------------------------------

        # ---- 1.1: En este caso, los obtengo aleatorios (en el ejemplo de la dnn están todas las opciones)
        xgboost_hyperparams: Dict[str, Union[int, float]] = HxXgboostHyperparameterBuilder().get_random_hyperparams()

        # --------------------------------------------------------------------------------------
        # -- 2: Instancio la clase que me va a proporcionar un toolkit para ese modelo en concreto
        # --------------------------------------------------------------------------------------

        # ---- 2.1: Le paso los hiperparametros deseados (Entrar en la clase para ver detalle)
        xgboost: HxXtremeGradientBoostingRegressor = HxXtremeGradientBoostingRegressor(
            data_dict=self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            hiperparams=xgboost_hyperparams,  # Diccionario con los hiperparametros del LGBM
            metric="mae",  # Literal['mse', 'rmse', 'mae', 'msle', 'rmsle', 'mape'], Metrica que vamos a evaluar
            problem_type="reg:absoluteerror",  # Literal['reg:quantileerror', 'reg:squarederror', 'reg:logistic', 'reg:squaredlogerror', 'reg:tweedie', 'reg:absoluteerror']
            save_path=None,  # Path donde se almacenarán los resultados (por defecto None, ya que se crean internamente las estructuras de carpetas)
            bins=self.bins,  # dict | None = None (Diccionario de estratificacion)
            verbose="dont_show",  # Literal['show_all', 'show_basics', 'dont_show'] = 'dont_show' (Cantidad de informacion adicional que se muestra en consola)
            random_state=42,  # Random state de incializacion de pesos para garantizar reproducibilidad

        )

        # ---- 2.2: [Metodo]: Permite entrenar el modelo, pintar las metricas en pantalla y devolver un diccionario de las metricas
        # xgboost.fit_and_get_metrics(save_metrics_dict=False)

        # ---- 2.3: [Metodo]: Metodo principal de la clase que ejecuta internamente 'fit_and_get_metrics', almacena el modelo junto con graficos e info adicional
        xgboost.fit_and_get_model_and_results()

        # ---- 2.4: [Metodo]: Permite ejecutar el analisis SHAP del modelo
        xgboost.execute_shap_analysis(
            sample=True,  # Realizar un sampleo de la data antes de calcular SHAP (!! Es muy intensivo en cómputo si no se samplea !!)
            num_features_to_show=100,  # En los gráficos se van a mostrar las N características mas importantes (con mas peso)
            num_sample=200,  # Número de instancias para background (explicador deep/grad).
        )

    def _cat_boost_regressor(self):
        """
        Metodo para realizar el entrenamiento del catboost de regresion
        :return:
        """

        # --------------------------------------------------------------------------------------
        # -- 1: Obtengo los hiperparametros que le voy a pasar
        # --------------------------------------------------------------------------------------

        # ---- 1.1: En este caso, los obtengo aleatorios (en el ejemplo de la dnn están todas las opciones)
        catboost_hyperparams: Dict[str, Union[int, float]] = HxCatBoostHyperparameterBuilder().get_random_hyperparams()

        # --------------------------------------------------------------------------------------
        # -- 2: Instancio la clase que me va a proporcionar un toolkit para ese modelo en concreto
        # --------------------------------------------------------------------------------------

        # ---- 2.1: Le paso los hiperparametros deseados (Entrar en la clase para ver detalle)
        catboost: HxCatBoostRegressor = HxCatBoostRegressor(
            data_dict=self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            hiperparams=catboost_hyperparams,  # Diccionario con los hiperparametros del LGBM
            metric="mae",  # Literal['mse', 'rmse', 'mae', 'msle', 'rmsle', 'mape'], Metrica que vamos a evaluar
            problem_type="MAE",  # Literal['RMSE', 'MAE', 'Quantile', 'Tweedie'] = 'MAE'
            save_path=None,  # Path donde se almacenarán los resultados (por defecto None, ya que se crean internamente las estructuras de carpetas)
            bins=self.bins,  # dict | None = None (Diccionario de estratificacion)
            verbose="dont_show",  # Literal['show_all', 'show_basics', 'dont_show'] = 'dont_show' (Cantidad de informacion adicional que se muestra en consola)
            random_state=42,  # Random state de incializacion de pesos para garantizar reproducibilidad

        )

        # ---- 2.2: [Metodo]: Permite entrenar el modelo, pintar las metricas en pantalla y devolver un diccionario de las metricas
        # catboost.fit_and_get_metrics(save_metrics_dict=False)

        # ---- 2.3: [Metodo]: Metodo principal de la clase que ejecuta internamente 'fit_and_get_metrics', almacena el modelo junto con graficos e info adicional
        catboost.fit_and_get_model_and_results()

        # ---- 2.4: [Metodo]: Permite ejecutar el analisis SHAP del modelo
        catboost.execute_shap_analysis(
            sample=True,  # Realizar un sampleo de la data antes de calcular SHAP (!! Es muy intensivo en cómputo si no se samplea !!)
            num_features_to_show=100,  # En los gráficos se van a mostrar las N características mas importantes (con mas peso)
            num_sample=200,  # Número de instancias para background (explicador deep/grad).
        )

    # </editor-fold>


class CaliforniaHousingExample:
    def __init__(self):
        """
        Ejemplo de regresión de modelos de ML y DL para el dataset California Housing de sklearn
        """

        # --------------------------------------------------------------------------------------
        # -- 0: Instancio InfoTools y pinto la entrada
        # --------------------------------------------------------------------------------------

        # ---- 0.1: Instancia de InfoTools
        self.IT: InfoTools = InfoTools()

        # ---- 0.2: Pinto la entrada
        self.IT.header_print(f"CaliforniaHousingExample: Ejemplo de regresión con sklearn.datasets.fetch_california_housing utilizando hx_ml_dl_tools")

        # --------------------------------------------------------------------------------------
        # -- 1: Descargo el dataset y obtengo el data dict
        # --------------------------------------------------------------------------------------

        # ---- 1.1: Descargo el dataset
        self.california_dataset = fetch_california_housing()

        # ---- 1.2: Transformo a dataframe (Lo requieren los modelos)
        self.x = pd.DataFrame(self.california_dataset.data, columns=self.california_dataset.feature_names)
        self.y = pd.DataFrame(self.california_dataset.target, columns=["target"])

        # ---- 1.3: Preprocesamiento - Escalar características (IMPORTANTE para redes neuronales)
        scaler = StandardScaler()
        self.x_scaled = pd.DataFrame(scaler.fit_transform(self.x), columns=self.x.columns)

        # ---- 1.4: Divido en train y test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_scaled, self.y, test_size=0.2, random_state=42)

        # ---- 1.5: Creo el diccionario en el formato requerido por los modelos
        self.data_dict = {"x_train": self.x_train, "y_train": self.y_train, "x_test": self.x_test, "y_test": self.y_test}

        # --------------------------------------------------------------------------------------
        # -- 2: Creo a mano el diccionario de bins para estratificar (adaptado a California Housing)
        # --------------------------------------------------------------------------------------

        # ---- 2.1: Bins específicos para los precios de viviendas de California (en $100,000)
        self.bins: dict = {
            "bins": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "labels": ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10"]
        }

        # ---- 2.2: Información del dataset
        self.IT.sub_intro_print(f"Dataset California Housing cargado:")
        self.IT.info_print(f"  - Muestras totales: {self.x.shape[0]}")
        self.IT.info_print(f"  - Características: {self.x.shape[1]}")
        self.IT.info_print(f"  - Rango target: {self.y['target'].min():.2f} - {self.y['target'].max():.2f} (en $100,000)")
        self.IT.info_print(f"  - Train samples: {self.x_train.shape[0]}")
        self.IT.info_print(f"  - Test samples: {self.x_test.shape[0]}")

        # ---- 2.3: Informacion de distribucion
        self.get_dataset_info()

    def execute_example(self):
        """
        Meodo que ejecuta en cadena los métodos de entrenamiento
        :return:
        """

        self.IT.sub_intro_print("Iniciando entrenamiento de modelos para California Housing")

        # Ejecutar modelos (puedes comentar los que no quieras ejecutar)
        self._light_gbm_regressor()
        self._xg_boost_regressor()
        self._cat_boost_regressor()
        self._dnn_regressor()

        self.IT.sub_intro_print("Entrenamiento completado para todos los modelos")

    def get_dataset_info(self):
        """
        Metodo adicional para obtener información detallada del dataset
        """
        self.IT.sub_intro_print("Estadísticas del target:")

        self.IT.info_print(f"Media: {self.y['target'].mean():.2f}")
        self.IT.info_print(f"Mediana: {self.y['target'].median():.2f}")
        self.IT.info_print(f"Std: {self.y['target'].std():.2f}")
        self.IT.info_print(f"Min: {self.y['target'].min():.2f}")
        self.IT.info_print(f"Max: {self.y['target'].max():.2f}")

    # <editor-fold desc="Modelos   -----------------------------------------------------------------------------------------------------------------------------------------------">

    def _dnn_regressor(self):
        """
        Metodo para realizar el entrenamiento de la red neuronal densa de regresión
        :return:
        """

        self.IT.sub_intro_print("Entrenando Red Neuronal Densa (DNN)")

        # --------------------------------------------------------------------------------------
        # -- 1: Obtengo los hiperparámetros optimizados para California Housing
        # --------------------------------------------------------------------------------------

        # ---- 1.1: Hiperparámetros específicos para este dataset más grande
        dnn_hyperparams = HxDnnHyperparameterBuilder({
            'units': 64,  # Más unidades para dataset más grande
            'dropout': 0.3,  # Dropout moderado
            'learning_rate': 0.001,  # Learning rate estándar
            'regularization': 0.01,  # Regularización ligera
            'epochs': 150  # Más épocas para convergencia
        }).validate_and_get_hyperparams()

        # --------------------------------------------------------------------------------------
        # -- 2: Instancio la clase DNN
        # --------------------------------------------------------------------------------------

        dnn: HxDenseNeuralNetworkRegressor = HxDenseNeuralNetworkRegressor(
            self.data_dict,  # Diccionario con los datos
            dnn_hyperparams,  # Hiperparámetros optimizados
            'mse',  # Métrica principal
            'regression',  # Tipo de problema
            2,  # Arquitectura "<>" (hourglass)
            3,  # 3 capas ocultas
            "relu",  # Función de activación
            "l2",  # Regularización L2
            "adam",  # Optimizador
            "mse",  # Función de pérdida
            True,  # Incluir dropout
            True,  # Incluir batch normalization
            20,  # Paciencia para early stopping
            None,  # Path de guardado (automático)
            self.bins,  # Bins para estratificación
            "dont_show",  # Verbose level
            42  # Random state
        )

        # ---- 2.2: Entrenar y obtener resultados completos
        dnn.fit_and_get_model_and_results()

        # ---- 2.3: Análisis SHAP (opcional - puede ser lento)
        dnn.execute_shap_analysis(sample=True, num_features_to_show=20, num_sample=50, background_sample=100)

    def _light_gbm_regressor(self):
        """
        Metodo para realizar el entrenamiento del LGBM de regresión
        :return:
        """

        self.IT.sub_intro_print("Entrenando LightGBM Regressor")

        # --------------------------------------------------------------------------------------
        # -- 1: Hiperparámetros para LightGBM
        # --------------------------------------------------------------------------------------

        lgbm_hyperparams: Dict[str, Union[int, float]] = HxLightgbmHyperparameterBuilder({
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }).validate_and_get_hyperparams()

        # --------------------------------------------------------------------------------------
        # -- 2: Instancio LightGBM
        # --------------------------------------------------------------------------------------

        lgbm: HxLightGbmRegressor = HxLightGbmRegressor(
            data_dict=self.data_dict,
            hiperparams=lgbm_hyperparams,
            metric="mse",
            problem_type="regression",
            save_path=None,
            bins=self.bins,
            verbose="show_basics",
            random_state=42
        )

        # ---- 2.2: Entrenar y obtener resultados
        lgbm.fit_and_get_model_and_results()

        # ---- 2.3: SHAP analysis
        lgbm.execute_shap_analysis(sample=True, num_features_to_show=20, num_sample=200)

    def _xg_boost_regressor(self):
        """
        Metodo para realizar el entrenamiento del XGBoost de regresión
        :return:
        """

        self.IT.sub_intro_print("Entrenando XGBoost Regressor")

        # --------------------------------------------------------------------------------------
        # -- 1: Hiperparámetros para XGBoost
        # --------------------------------------------------------------------------------------

        xgboost_hyperparams: Dict[str, Union[int, float]] = HxXgboostHyperparameterBuilder({
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.8
        }).validate_and_get_hyperparams()

        # --------------------------------------------------------------------------------------
        # -- 2: Instancio XGBoost
        # --------------------------------------------------------------------------------------

        xgboost: HxXtremeGradientBoostingRegressor = HxXtremeGradientBoostingRegressor(
            data_dict=self.data_dict,
            hiperparams=xgboost_hyperparams,
            metric="mse",
            problem_type="reg:squarederror",
            save_path=None,
            bins=self.bins,
            verbose="show_basics",
            random_state=42
        )

        # ---- 2.2: Entrenar y obtener resultados
        xgboost.fit_and_get_model_and_results()

        # ---- 2.3: SHAP analysis
        xgboost.execute_shap_analysis(sample=True, num_features_to_show=20, num_sample=200)

    def _cat_boost_regressor(self):
        """
        Metodo para realizar el entrenamiento del CatBoost de regresión
        :return:
        """

        self.IT.sub_intro_print("Entrenando CatBoost Regressor")

        # --------------------------------------------------------------------------------------
        # -- 1: Hiperparámetros para CatBoost
        # --------------------------------------------------------------------------------------

        catboost_hyperparams: Dict[str, Union[int, float]] = HxCatBoostHyperparameterBuilder({
            'iterations': 200,
            'depth': 6,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3
        }).validate_and_get_hyperparams()

        # --------------------------------------------------------------------------------------
        # -- 2: Instancio CatBoost
        # --------------------------------------------------------------------------------------

        catboost: HxCatBoostRegressor = HxCatBoostRegressor(
            data_dict=self.data_dict,
            hiperparams=catboost_hyperparams,
            metric="mse",
            problem_type="RMSE",
            save_path=None,
            bins=self.bins,
            verbose="show_basics",
            random_state=42
        )

        # ---- 2.2: Entrenar y obtener resultados
        catboost.fit_and_get_model_and_results()

        # ---- 2.3: SHAP analysis
        catboost.execute_shap_analysis(sample=True, num_features_to_show=20, num_sample=200)

    # </editor-fold>
