from hx_machine_learning_tools import HxLightGbmBinaryClassifier, HxXtremeGradientBoostingBinaryClassifier, HxCatBoostBinaryClassifier
from hx_deep_learning_tools import HxDenseNeuralNetworkBinaryClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from hx_hyperparameters_builder import *
from info_tools import InfoTools
import pandas as pd
import numpy as np


class BreastCancerExample:
    def __init__(self):
        """
        Ejemplo de clasificacion binaria de modelos de ML y DL para el ejemplo de la diabetes de sklearn
        """

        # --------------------------------------------------------------------------------------
        # -- 0: Instancio InfoTools y pinto la entrada
        # --------------------------------------------------------------------------------------

        # ---- 0.1: Instancia de InfoTools
        self.IT: InfoTools = InfoTools()

        # ---- 0.2: Pinto la entrada
        self.IT.header_print(f"DiabetesExample: Ejemplo de clasificacion con sklearn.datasets.breast_cancer utilizando hx_ml_dl_tools")

        # --------------------------------------------------------------------------------------
        # -- 1: Descargo el dataset y obtengo el data dict
        # --------------------------------------------------------------------------------------

        # ---- 1.1: Descargo el dataset
        self.cancer_dataset = load_breast_cancer()

        # ---- 1.2: Transformo a dataframe (Lo requieren los modelos)
        self.x = pd.DataFrame(self.cancer_dataset.data, columns=self.cancer_dataset.feature_names)
        self.y = pd.DataFrame(self.cancer_dataset.target, columns=["target"])

        # ---- 1.3: Divido en train y test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        # ---- 1.4: Creo el diccionario en el formato requerido por los modelos
        self.data_dict = {"x_train": self.x_train, "y_train": self.y_train, "x_test": self.x_test, "y_test": self.y_test}

        # --------------------------------------------------------------------------------------
        # -- 2: Obtengo informacion basica del dataset
        # --------------------------------------------------------------------------------------

        # ---- 2.1: Información del dataset
        self.IT.sub_intro_print(f"Dataset BreastCancer cargado:")
        self.IT.info_print(f"Muestras totales: {self.x.shape[0]}")
        self.IT.info_print(f"Características: {self.x.shape[1]}")
        self.IT.info_print(f"Rango target: {self.y['target'].min():.2f} - {self.y['target'].max():.2f}")
        self.IT.info_print(f"Train samples: {self.x_train.shape[0]}")
        self.IT.info_print(f"Test samples: {self.x_test.shape[0]}")

        # ---- 2.2: Informacion de distribucion
        self.get_dataset_info()

    def execute_example(self):
        """
        Metodo que ejecuta en cadena los metodos de entrenamiento
        :return:
        """

        self._light_gbm_classifier()
        self._xg_boost_classifier()
        self._cat_boost_classifier()
        self._dnn_binary_calssifier()

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

    def _dnn_binary_calssifier(self):
        """
        Metodo para realizar el entrenamiento de la red neuronal densa de clasificacion binaria
        :return:
        """

        # --------------------------------------------------------------------------------------
        # -- 1: Obtengo los hiperparametros que le voy a pasar
        # --------------------------------------------------------------------------------------

        # ---- 1.1: Opcion de pasarselos a mano
        # dnn_hyperparams: Dict[str, Union[int, float]] = {'units': 8, 'dropout': 0.5, 'learning_rate': 0.01, 'regularization': 0.1,  'epochs': 200}

        # ---- 1.2: Opcion de usar la clase HxDnnHyperparameterBuilder

        # ------ 1.2.1: Obtener parámetros aleatorios dentro del rango preestablecido en HxDnnHyperparameterBuilder
        # dnn_hyperparams: Dict[str, Union[int, float]] = HxDnnHyperparameterBuilder().get_random_hyperparams()

        # ------ 1.2.2: Pasarle los hiperparametros al constructor para que valide si hay alguno que se usa poco o no existe (USAMOS ESTA PARA EL EJEMPLO)
        dnn_hyperparams = HxDnnHyperparameterBuilder({'units': 8, 'dropout': 0.5, 'learning_rate': 0.01, 'regularization': 0.1,  'epochs': 200}).validate_and_get_hyperparams()

        # --------------------------------------------------------------------------------------
        # -- 2: Instancio la clase que me va a proporcionar un toolkit para ese modelo en concreto
        # --------------------------------------------------------------------------------------

        # ---- 2.1: Le paso los hiperparametros deseados (Entrar en la clase para ver detalle)
        dnn: HxDenseNeuralNetworkBinaryClassifier = HxDenseNeuralNetworkBinaryClassifier(
            self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            dnn_hyperparams,  # Diccionario de hiperparametros
            'accuracy',  # Literal['accuracy', 'precision', 'recall', 'f1', 'specificity', 'balanced_acc', 'roc_auc']
            'binary',  # Literal['binary', 'multiclass']
            1,  # Literal[0, 1, 2, 3] = 1 ({0: "<", 1: ">", 2: "<>", 3: "><"}) Arquitectura de la red
            1,  # Capas ocultas de la red (Toda red tiene 1 capa de entrada + N capas ocultas + 1 capa de salida)
            "relu",  # Literal['relu', 'tanh', 'sigmoid'] = 'relu' (Funcion de activacion)
            "l1",  # Literal['l1', 'l2'] = 'l2'  (Tipo de regularizacion que se aplicará en caso de aplicar regularizacion)
            "adam",  # Literal['sgd', 'adam', 'rmsprop'] = 'adam'  (Optimizador que se va a utilizar)
            True,  # bool (Incluir o no capas de dropout (en caso de no incluirse, el hiperparametro dropout no hace nada))
            True,  # bool (Incluir capa de batch normalization)
            15,  # Si en N epochs no ha mejorado la métrica, aplica un earlyStop callback y devuelve el mejor modelo hasta la fecha
            None,  # Path donde se almacenarán los resultados (por defecto None, ya que se crean internamente las estructuras de carpetas)
            None,  #  dict | None = None (Diccionario de estratificacion, en binary está por defecto entre 0 y 1)
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

    def _light_gbm_classifier(self):
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
        lgbm: HxLightGbmBinaryClassifier = HxLightGbmBinaryClassifier(
            data_dict=self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            hiperparams=lgbm_hyperparams,  # Diccionario con los hiperparametros del LGBM
            metric="roc_auc",  # Literal['accuracy', 'precision', 'recall', 'f1', 'specificity', 'balanced_acc', 'roc_auc'], Metrica que vamos a evaluar
            problem_type="binary",  # Literal['binary', 'multiclass']
            save_path=None,  # Path donde se almacenarán los resultados (por defecto None, ya que se crean internamente las estructuras de carpetas)
            bins=None,  # En binary viene por defecto
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

    def _xg_boost_classifier(self):
        """
        Metodo para realizar el entrenamiento del xgboost de clasificacion
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
        xgboost: HxXtremeGradientBoostingBinaryClassifier = HxXtremeGradientBoostingBinaryClassifier(
            data_dict=self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            hiperparams=xgboost_hyperparams,  # Diccionario con los hiperparametros del XGB
            metric="roc_auc",  # Literal['accuracy', 'precision', 'recall', 'f1', 'specificity', 'balanced_acc', 'roc_auc'], Metrica que vamos a evaluar
            problem_type="binary",  # Literal['binary', 'multiclass']
            save_path=None,  # Path donde se almacenarán los resultados (por defecto None, ya que se crean internamente las estructuras de carpetas)
            bins=None,  # En binary viene por defecto
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

    def _cat_boost_classifier(self):
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
        catboost: HxCatBoostBinaryClassifier = HxCatBoostBinaryClassifier(
            data_dict=self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            hiperparams=catboost_hyperparams,  # Diccionario con los hiperparametros del XGB
            metric="roc_auc",  # Literal['accuracy', 'precision', 'recall', 'f1', 'specificity', 'balanced_acc', 'roc_auc'], Metrica que vamos a evaluar
            problem_type="binary",  # Literal['binary', 'multiclass']
            save_path=None,  # Path donde se almacenarán los resultados (por defecto None, ya que se crean internamente las estructuras de carpetas)
            bins=None,  # En binary viene por defecto
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


class CreditFraudExample:
    def __init__(self, sample_size: int = 100000):
        """
        Ejemplo de clasificación binaria con dataset de Fraud Detection (más grande y profesional)
        Dataset: Credit Card Fraud Detection from Kaggle (via OpenML)
        """

        # --------------------------------------------------------------------------------------
        # -- 0: Instancio InfoTools y pinto la entrada
        # --------------------------------------------------------------------------------------

        self.IT: InfoTools = InfoTools()
        self.IT.header_print(f"CreditFraudExample: Clasificación binaria con dataset profesional de Fraud Detection")

        # --------------------------------------------------------------------------------------
        # -- 1: Descargo el dataset de Fraud Detection (más grande y realista)
        # --------------------------------------------------------------------------------------

        self.IT.info_print("Cargando dataset de Credit Card Fraud Detection...")

        try:
            # Dataset de fraud detection (más de 280,000 transacciones)
            fraud_data = fetch_openml(name='creditcard', version=1, as_frame=True)
            self.x = fraud_data.data
            self.y = fraud_data.target

            # Si el target es string, convertirlo a numérico
            if self.y.dtype == 'object':
                self.y = self.y.astype(int)

        except Exception as e:
            self.IT.info_print(f"Error cargando dataset de fraud: {e}")
            self.IT.info_print("Cargando dataset alternativo Adult Income...")
            # Fallback a Adult Income dataset
            adult_data = fetch_openml(name='adult', version=2, as_frame=True)
            self.x = adult_data.data
            self.y = adult_data.target
            # Codificar target binario
            le = LabelEncoder()
            self.y = pd.Series(le.fit_transform(self.y))

        # --------------------------------------------------------------------------------------
        # -- 2: Preprocesamiento para dataset grande - CORRECCIÓN CRÍTICA
        # --------------------------------------------------------------------------------------

        # ---- 2.1: Asegurar que x e y tengan el mismo índice
        self.x = self.x.reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)

        # ---- 2.2: Samplear si el dataset es muy grande para pruebas rápidas
        if len(self.x) > sample_size:
            self.IT.info_print(f"Realizando sampleo de {sample_size} muestras...")
            indices = np.random.choice(len(self.x), sample_size, replace=False)
            self.x = self.x.iloc[indices].reset_index(drop=True)
            self.y = self.y.iloc[indices].reset_index(drop=True)

        # ---- 2.3: Limpiar y preprocesar
        self.x = self._clean_dataframe(self.x)

        # ---- 2.4: Escalar características (CRÍTICO para redes neuronales)
        scaler = StandardScaler()
        self.x_scaled = pd.DataFrame(scaler.fit_transform(self.x), columns=self.x.columns)

        # ---- 2.5: Convertir y a DataFrame - CORRECCIÓN: asegurar mismo índice
        self.y_df = pd.DataFrame(self.y.values, columns=["target"])  # Usar .values en lugar de la serie directamente
        self.y_df.index = self.x_scaled.index  # Asegurar mismos índices

        # ---- 2.6: Verificar que las dimensiones coincidan
        self.IT.info_print(f"Verificando dimensiones: X_scaled {self.x_scaled.shape}, y_df {self.y_df.shape}")

        if len(self.x_scaled) != len(self.y_df):
            self.IT.info_print("Ajustando dimensiones...")
            # Tomar el mínimo común de muestras
            min_len = min(len(self.x_scaled), len(self.y_df))
            self.x_scaled = self.x_scaled.iloc[:min_len]
            self.y_df = self.y_df.iloc[:min_len]

        # ---- 2.7: Dividir en train y test con estratificación
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_scaled, self.y_df, test_size=0.2, random_state=42, stratify=self.y_df
        )

        # ---- 2.8: Crear data_dict
        self.data_dict = {
            "x_train": self.x_train,
            "y_train": self.y_train,
            "x_test": self.x_test,
            "y_test": self.y_test
        }

        # --------------------------------------------------------------------------------------
        # -- 3: Información del dataset
        # --------------------------------------------------------------------------------------

        self.IT.sub_intro_print(f"Dataset Profesional Cargado:")
        self.IT.info_print(f"Muestras totales: {self.x.shape[0]:,}")
        self.IT.info_print(f"Características: {self.x.shape[1]}")

        class_info = self.y_df['target'].value_counts()
        total = len(self.y_df)
        self.IT.info_print(f"Clases: {class_info.to_dict()}")
        self.IT.info_print(f"Proporciones: {({k: f'{v / total * 100:.1f}%' for k, v in class_info.items()})}")

        self.IT.info_print(f"Train samples: {self.x_train.shape[0]:,}")
        self.IT.info_print(f"Test samples: {self.x_test.shape[0]:,}")

        self.get_dataset_info()

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Limpieza básica del dataframe"""
        # Eliminar columnas con demasiados missing values
        df = df.dropna(axis=1, thresh=len(df) * 0.8)

        # Llenar valores missing con la mediana para numéricos y moda para categóricos
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')

        # Convertir variables categóricas a numéricas si las hay
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            le = LabelEncoder()
            for col in categorical_cols:
                df[col] = le.fit_transform(df[col].astype(str))

        return df

    def execute_example(self):
        """
        Método que ejecuta en cadena los métodos de entrenamiento
        """
        self.IT.sub_intro_print("Iniciando entrenamiento de modelos para Fraud Detection")

        # Ejecutar modelos (comenta los que no quieras ejecutar)
        self._light_gbm_classifier()
        self._xg_boost_classifier()
        self._cat_boost_classifier()
        self._dnn_binary_classifier()

        self.IT.sub_intro_print("Entrenamiento completado para todos los modelos")

    def get_dataset_info(self):
        """Información detallada del dataset"""
        self.IT.sub_intro_print("Análisis Detallado del Dataset:")

        class_counts = self.y_df['target'].value_counts()
        class_percentages = (class_counts / len(self.y_df)) * 100

        self.IT.info_print("Distribución de clases:")
        for class_val, count in class_counts.items():
            percentage = class_percentages[class_val]
            class_name = "Fraude" if class_val == 1 else "Normal" if 'creditcard' in str(self.x.columns) else "Clase 0/1"
            self.IT.info_print(f"  - {class_name} (Clase {class_val}): {count:,} muestras ({percentage:.2f}%)")

        self.IT.info_print(f"Balance: {class_percentages[0]:.2f}% vs {class_percentages[1]:.2f}%")
        self.IT.info_print(f"Dimensiones: {self.x.shape[0]:,} filas × {self.x.shape[1]} columnas")

    # <editor-fold desc="Modelos -------------------------------------------------------------------------------------------------------------------------------------------------">

    def _dnn_binary_classifier(self):
        """
        Metodo para realizar el entrenamiento de la red neuronal densa de clasificacion binaria
        :return:
        """

        # --------------------------------------------------------------------------------------
        # -- 1: Obtengo los hiperparametros que le voy a pasar
        # --------------------------------------------------------------------------------------

        # ---- 1.1: Opcion de pasarselos a mano
        # dnn_hyperparams: Dict[str, Union[int, float]] = {'units': 8, 'dropout': 0.5, 'learning_rate': 0.01, 'regularization': 0.1,  'epochs': 200}

        # ---- 1.2: Opcion de usar la clase HxDnnHyperparameterBuilder

        # ------ 1.2.1: Obtener parámetros aleatorios dentro del rango preestablecido en HxDnnHyperparameterBuilder
        # dnn_hyperparams: Dict[str, Union[int, float]] = HxDnnHyperparameterBuilder().get_random_hyperparams()

        # ------ 1.2.2: Pasarle los hiperparametros al constructor para que valide si hay alguno que se usa poco o no existe (USAMOS ESTA PARA EL EJEMPLO)
        dnn_hyperparams = HxDnnHyperparameterBuilder({
            'units': 128,  # Más unidades para dataset grande
            'dropout': 0.3,  # Dropout moderado
            'learning_rate': 0.001,  # Learning rate estándar
            'regularization': 0.001,  # Regularización ligera
            'epochs': 50  # Menos épocas para dataset grande
        }).validate_and_get_hyperparams()

        # --------------------------------------------------------------------------------------
        # -- 2: Instancio la clase que me va a proporcionar un toolkit para ese modelo en concreto
        # --------------------------------------------------------------------------------------

        # ---- 2.1: Le paso los hiperparametros deseados (Entrar en la clase para ver detalle)
        dnn: HxDenseNeuralNetworkBinaryClassifier = HxDenseNeuralNetworkBinaryClassifier(
            self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            dnn_hyperparams,  # Diccionario de hiperparametros
            'accuracy',  # Literal['accuracy', 'precision', 'recall', 'f1', 'specificity', 'balanced_acc', 'roc_auc']
            'binary',  # Literal['binary', 'multiclass']
            1,  # Literal[0, 1, 2, 3] = 1 ({0: "<", 1: ">", 2: "<>", 3: "><"}) Arquitectura de la red
            1,  # Capas ocultas de la red (Toda red tiene 1 capa de entrada + N capas ocultas + 1 capa de salida)
            "relu",  # Literal['relu', 'tanh', 'sigmoid'] = 'relu' (Funcion de activacion)
            "l2",  # Literal['l1', 'l2'] = 'l2'  (Tipo de regularizacion que se aplicará en caso de aplicar regularizacion)
            "adam",  # Literal['sgd', 'adam', 'rmsprop'] = 'adam'  (Optimizador que se va a utilizar)
            True,  # bool (Incluir o no capas de dropout (en caso de no incluirse, el hiperparametro dropout no hace nada))
            True,  # bool (Incluir capa de batch normalization)
            15,  # Si en N epochs no ha mejorado la métrica, aplica un earlyStop callback y devuelve el mejor modelo hasta la fecha
            None,  # Path donde se almacenarán los resultados (por defecto None, ya que se crean internamente las estructuras de carpetas)
            None,  #  dict | None = None (Diccionario de estratificacion, en binary está por defecto entre 0 y 1)
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
            num_sample=50,  # Cantidad de muestra (filas) con las que se va a realizar el analisis
            background_sample=50,  # Número de instancias para background (explicador deep/grad).
        )

    def _light_gbm_classifier(self):
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
        lgbm: HxLightGbmBinaryClassifier = HxLightGbmBinaryClassifier(
            data_dict=self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            hiperparams=lgbm_hyperparams,  # Diccionario con los hiperparametros del LGBM
            metric="roc_auc",  # Literal['accuracy', 'precision', 'recall', 'f1', 'specificity', 'balanced_acc', 'roc_auc'], Metrica que vamos a evaluar
            problem_type="binary",  # Literal['binary', 'multiclass']
            save_path=None,  # Path donde se almacenarán los resultados (por defecto None, ya que se crean internamente las estructuras de carpetas)
            bins=None,  # En binary viene por defecto
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

    def _xg_boost_classifier(self):
        """
        Metodo para realizar el entrenamiento del xgboost de clasificacion
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
        xgboost: HxXtremeGradientBoostingBinaryClassifier = HxXtremeGradientBoostingBinaryClassifier(
            data_dict=self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            hiperparams=xgboost_hyperparams,  # Diccionario con los hiperparametros del XGB
            metric="roc_auc",  # Literal['accuracy', 'precision', 'recall', 'f1', 'specificity', 'balanced_acc', 'roc_auc'], Metrica que vamos a evaluar
            problem_type="binary",  # Literal['binary', 'multiclass']
            save_path=None,  # Path donde se almacenarán los resultados (por defecto None, ya que se crean internamente las estructuras de carpetas)
            bins=None,  # En binary viene por defecto
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

    def _cat_boost_classifier(self):
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
        catboost: HxCatBoostBinaryClassifier = HxCatBoostBinaryClassifier(
            data_dict=self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            hiperparams=catboost_hyperparams,  # Diccionario con los hiperparametros del XGB
            metric="roc_auc",  # Literal['accuracy', 'precision', 'recall', 'f1', 'specificity', 'balanced_acc', 'roc_auc'], Metrica que vamos a evaluar
            problem_type="binary",  # Literal['binary', 'multiclass']
            save_path=None,  # Path donde se almacenarán los resultados (por defecto None, ya que se crean internamente las estructuras de carpetas)
            bins=None,  # En binary viene por defecto
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



