from hx_ml_dl_tools import HxLightGbmMulticlassClassifier, HxXtremeGradientBoostingMulticlassClassifier, HxCatBoostMulticlassClassifier, HxDenseNeuralNetworkMulticlassClassifier
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from hx_hyperparameters_builder import HxLightgbmHyperparameterBuilder, HxXgboostHyperparameterBuilder, HxCatBoostHyperparameterBuilder, HxDnnHyperparameterBuilder
from typing import Dict, Union
from info_tools import InfoTools
import pandas as pd
import numpy as np


class IrisMulticlassExample:
    def __init__(self):
        """
        Ejemplo de clasificación MULTICLASE de modelos de ML para el dataset Iris de sklearn
        """

        # --------------------------------------------------------------------------------------
        # -- 0: Instancio InfoTools y pinto la entrada
        # --------------------------------------------------------------------------------------

        # ---- 0.1: Instancia de InfoTools
        self.IT: InfoTools = InfoTools()

        # ---- 0.2: Pinto la entrada
        self.IT.header_print(f"IrisMulticlassExample: Ejemplo de clasificación MULTICLASE con sklearn.datasets.iris utilizando hx_ml_dl_tools")

        # --------------------------------------------------------------------------------------
        # -- 1: Descargo el dataset multiclase y obtengo el data dict
        # --------------------------------------------------------------------------------------

        # ---- 1.1: Descargo el dataset Iris (3 clases)
        self.iris_dataset = load_iris()

        # ---- 1.2: Transformo a dataframe (Lo requieren los modelos)
        self.x = pd.DataFrame(self.iris_dataset.data, columns=self.iris_dataset.feature_names)
        self.y = pd.DataFrame(self.iris_dataset.target, columns=["target"])

        # ---- 1.3: Divido en train y test con estratificación para multiclase
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # ---- 1.4: Creo el diccionario en el formato requerido por los modelos
        self.data_dict = {
            "x_train": self.x_train,
            "y_train": self.y_train,
            "x_test": self.x_test,
            "y_test": self.y_test
        }

        # ---- 1.5: Número de clases
        self.n_classes = len(np.unique(self.y))

        # --------------------------------------------------------------------------------------
        # -- 2: Obtengo informacion basica del dataset multiclase
        # --------------------------------------------------------------------------------------

        # ---- 2.1: Información del dataset
        self.IT.sub_intro_print(f"Dataset Iris (Multiclase) cargado:")
        self.IT.info_print(f"Muestras totales: {self.x.shape[0]}")
        self.IT.info_print(f"Características: {self.x.shape[1]}")
        self.IT.info_print(f"Número de clases: {self.n_classes}")
        self.IT.info_print(f"Clases: {self.iris_dataset.target_names.tolist()}")
        self.IT.info_print(f"Train samples: {self.x_train.shape[0]}")
        self.IT.info_print(f"Test samples: {self.x_test.shape[0]}")

        # ---- 2.2: Informacion de distribucion
        self.get_dataset_info()

    def execute_example(self):
        """
        Metodo que ejecuta el entrenamiento multiclase
        :return:
        """
        self._light_gbm_multiclass_classifier()
        self._xg_boost_multiclass_classifier()
        self._cat_boost_multiclass_classifier()
        self._dnn_multiclass_classifier()

    def get_dataset_info(self):
        """
        Metodo adicional para obtener información detallada del dataset multiclase
        """
        self.IT.sub_intro_print("Estadísticas del target (Multiclase):")

        # Distribución por clase
        class_counts = self.y['target'].value_counts().sort_index()
        class_percentages = (class_counts / len(self.y)) * 100

        self.IT.info_print("Distribución por clase:")
        for class_idx, count in class_counts.items():
            class_name = self.iris_dataset.target_names[class_idx]
            percentage = class_percentages[class_idx]
            self.IT.info_print(f"  - {class_name} (Clase {class_idx}): {count} muestras ({percentage:.1f}%)")

        self.IT.info_print(f"Balance: {'Balanceado' if class_percentages.std() < 10 else 'Desbalanceado'}")

        # Estadísticas generales
        self.IT.info_print(f"Media target: {self.y['target'].mean():.2f}")
        self.IT.info_print(f"Std target: {self.y['target'].std():.2f}")

    # <editor-fold desc="Modelo Multiclase   -------------------------------------------------------------------------------------------------------------------------------------">

    def _light_gbm_multiclass_classifier(self):
        """
        Metodo para realizar el entrenamiento del LGBM para clasificación MULTICLASE
        :return:
        """

        # --------------------------------------------------------------------------------------
        # -- 1: Obtengo los hiperparametros que le voy a pasar (adaptados para multiclase)
        # --------------------------------------------------------------------------------------

        # ---- 1.1: Hiperparámetros específicos para multiclase
        lgbm_hyperparams: Dict[str, Union[int, float]] = HxLightgbmHyperparameterBuilder().get_random_hyperparams()

        # --------------------------------------------------------------------------------------
        # -- 2: Instancio la clase MULTICLASE que me va a proporcionar un toolkit para ese modelo
        # --------------------------------------------------------------------------------------

        # ---- 2.1: Le paso los hiperparametros deseados para multiclase
        lgbm_multiclass: HxLightGbmMulticlassClassifier = HxLightGbmMulticlassClassifier(
            data_dict=self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            hiperparams=lgbm_hyperparams,  # Diccionario con los hiperparametros del LGBM
            metric="f1_macro",  # Métrica específica para multiclase
            problem_type="multiclass",  # Tipo de problema MULTICLASE
            save_path=None,  # Path donde se almacenarán los resultados
            bins=None,  # Bins para estratificación
            verbose="dont_show",  # Nivel de verbosidad
            random_state=42,  # Random state para reproducibilidad
            n_classes=self.n_classes  # Número de clases (CRÍTICO para multiclase)
        )

        # ---- 2.2: Metodo principal que ejecuta el entrenamiento completo
        lgbm_multiclass.fit_and_get_model_and_results()

        # ---- 2.3: Ejecutar análisis SHAP (opcional, puede ser intensivo computacionalmente)

        lgbm_multiclass.execute_shap_analysis(
            sample=True,  # Samplear para hacerlo más rápido
            num_features_to_show=10,  # Mostrar top 10 características
            num_sample=100  # Usar 100 muestras para el análisis
        )

    def _xg_boost_multiclass_classifier(self):
        """
        Metodo para realizar el entrenamiento del LGBM para clasificación MULTICLASE
        :return:
        """

        # --------------------------------------------------------------------------------------
        # -- 1: Obtengo los hiperparametros que le voy a pasar (adaptados para multiclase)
        # --------------------------------------------------------------------------------------

        # ---- 1.1: Hiperparámetros específicos para multiclase
        xgb_hyperparams: Dict[str, Union[int, float]] = HxXgboostHyperparameterBuilder().get_random_hyperparams()

        # --------------------------------------------------------------------------------------
        # -- 2: Instancio la clase MULTICLASE que me va a proporcionar un toolkit para ese modelo
        # --------------------------------------------------------------------------------------

        # ---- 2.1: Le paso los hiperparametros deseados para multiclase
        xgb_multiclass: HxXtremeGradientBoostingMulticlassClassifier = HxXtremeGradientBoostingMulticlassClassifier(
            data_dict=self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            hiperparams=xgb_hyperparams,  # Diccionario con los hiperparametros del LGBM
            metric="f1_macro",  # Métrica específica para multiclase
            problem_type="multiclass",  # Tipo de problema MULTICLASE
            save_path=None,  # Path donde se almacenarán los resultados
            bins=None,  # Bins para estratificación
            verbose="dont_show",  # Nivel de verbosidad
            random_state=42,  # Random state para reproducibilidad
            n_classes=self.n_classes  # Número de clases (CRÍTICO para multiclase)
        )

        # ---- 2.2: Metodo principal que ejecuta el entrenamiento completo
        xgb_multiclass.fit_and_get_model_and_results()

        # ---- 2.3: Ejecutar análisis SHAP (opcional, puede ser intensivo computacionalmente)

        xgb_multiclass.execute_shap_analysis(
            sample=True,  # Samplear para hacerlo más rápido
            num_features_to_show=10,  # Mostrar top 10 características
            num_sample=100  # Usar 100 muestras para el análisis
        )

    def _cat_boost_multiclass_classifier(self):
        """
        Metodo para realizar el entrenamiento del CAT para clasificación MULTICLASE
        :return:
        """

        # --------------------------------------------------------------------------------------
        # -- 1: Obtengo los hiperparametros que le voy a pasar (adaptados para multiclase)
        # --------------------------------------------------------------------------------------

        # ---- 1.1: Hiperparámetros específicos para multiclase
        cat_hyperparams: Dict[str, Union[int, float]] = HxCatBoostHyperparameterBuilder().get_random_hyperparams()

        # --------------------------------------------------------------------------------------
        # -- 2: Instancio la clase MULTICLASE que me va a proporcionar un toolkit para ese modelo
        # --------------------------------------------------------------------------------------

        # ---- 2.1: Le paso los hiperparametros deseados para multiclase
        cat_multiclass: HxCatBoostMulticlassClassifier = HxCatBoostMulticlassClassifier(
            data_dict=self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            hiperparams=cat_hyperparams,  # Diccionario con los hiperparametros del LGBM
            metric="f1_macro",  # Métrica específica para multiclase
            problem_type="multiclass",  # Tipo de problema MULTICLASE
            save_path=None,  # Path donde se almacenarán los resultados
            bins=None,  # Bins para estratificación
            verbose="dont_show",  # Nivel de verbosidad
            random_state=42,  # Random state para reproducibilidad
            n_classes=self.n_classes  # Número de clases (CRÍTICO para multiclase)
        )

        # ---- 2.2: Metodo principal que ejecuta el entrenamiento completo
        cat_multiclass.fit_and_get_model_and_results()

        # ---- 2.3: Ejecutar análisis SHAP (opcional, puede ser intensivo computacionalmente)

        """cat_multiclass.execute_shap_analysis(
            sample=True,  # Samplear para hacerlo más rápido
            num_features_to_show=10,  # Mostrar top 10 características
            num_sample=100  # Usar 100 muestras para el análisis
        )"""

    def _dnn_multiclass_classifier(self):
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
        dnn: HxDenseNeuralNetworkMulticlassClassifier = HxDenseNeuralNetworkMulticlassClassifier(
            self.data_dict,  # Diccionario con x_train, x_test, y_train, y_test
            dnn_hyperparams,  # Diccionario de hiperparametros
            'accuracy',  # ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_micro', 'balanced_accuracy', 'roc_auc_ovr']
            'multiclass',  # Literal['binary', 'multiclass']
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

    # </editor-fold>


class WineQualityMulticlassExample:
    def __init__(self):
        """
        Ejemplo de clasificación MULTICLASE con dataset de Vinos (sklearn)
        """
        # --------------------------------------------------------------------------------------
        # -- 0: Instancio InfoTools y pinto la entrada
        # --------------------------------------------------------------------------------------
        self.IT: InfoTools = InfoTools()
        self.IT.header_print(f"WineQualityMulticlassExample: Ejemplo de clasificación MULTICLASE con sklearn.datasets.wine utilizando hx_ml_dl_tools")

        # --------------------------------------------------------------------------------------
        # -- 1: Descargo el dataset multiclase y obtengo el data dict
        # --------------------------------------------------------------------------------------
        self.wine_dataset = load_wine()
        self.x = pd.DataFrame(self.wine_dataset.data, columns=self.wine_dataset.feature_names)
        self.y = pd.DataFrame(self.wine_dataset.target, columns=["target"])

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        self.data_dict = {
            "x_train": self.x_train,
            "y_train": self.y_train,
            "x_test": self.x_test,
            "y_test": self.y_test
        }
        self.n_classes = len(np.unique(self.y))

        # --------------------------------------------------------------------------------------
        # -- 2: Obtengo informacion basica del dataset multiclase
        # --------------------------------------------------------------------------------------
        self.IT.sub_intro_print(f"Dataset Wine (Multiclase) cargado:")
        self.IT.info_print(f"Muestras totales: {self.x.shape[0]}")
        self.IT.info_print(f"Características: {self.x.shape[1]}")
        self.IT.info_print(f"Número de clases: {self.n_classes}")
        self.IT.info_print(f"Clases: {self.wine_dataset.target_names.tolist()}")
        self.IT.info_print(f"Train samples: {self.x_train.shape[0]}")
        self.IT.info_print(f"Test samples: {self.x_test.shape[0]}")

        self.get_dataset_info()

    def execute_example(self):
        """
        Metodo que ejecuta el entrenamiento multiclase
        """
        self._light_gbm_multiclass_classifier()
        self._xg_boost_multiclass_classifier()
        self._cat_boost_multiclass_classifier()
        self._dnn_multiclass_classifier()

    def get_dataset_info(self):
        """
        Metodo adicional para obtener información detallada del dataset multiclase
        """
        self.IT.sub_intro_print("Estadísticas del target (Multiclase):")
        class_counts = self.y['target'].value_counts().sort_index()
        class_percentages = (class_counts / len(self.y)) * 100

        self.IT.info_print("Distribución por clase:")
        for class_idx, count in class_counts.items():
            class_name = self.wine_dataset.target_names[class_idx]
            percentage = class_percentages[class_idx]
            self.IT.info_print(f"  - {class_name} (Clase {class_idx}): {count} muestras ({percentage:.1f}%)")

        self.IT.info_print(f"Balance: {'Balanceado' if class_percentages.std() < 10 else 'Desbalanceado'}")
        self.IT.info_print(f"Media target: {self.y['target'].mean():.2f}")
        self.IT.info_print(f"Std target: {self.y['target'].std():.2f}")

    # <editor-fold desc="Modelo Multiclase ------------------------------------------------------------------------------------------------------------------------------------->

    def _light_gbm_multiclass_classifier(self):
        lgbm_hyperparams = HxLightgbmHyperparameterBuilder().get_random_hyperparams()
        lgbm_multiclass = HxLightGbmMulticlassClassifier(
            data_dict=self.data_dict, hiperparams=lgbm_hyperparams, metric="f1_macro",
            problem_type="multiclass", save_path=None, bins=None, verbose="dont_show",
            random_state=42, n_classes=self.n_classes
        )
        lgbm_multiclass.fit_and_get_model_and_results()
        lgbm_multiclass.execute_shap_analysis(sample=True, num_features_to_show=10, num_sample=100)

    def _xg_boost_multiclass_classifier(self):
        xgb_hyperparams = HxXgboostHyperparameterBuilder().get_random_hyperparams()
        xgb_multiclass = HxXtremeGradientBoostingMulticlassClassifier(
            data_dict=self.data_dict, hiperparams=xgb_hyperparams, metric="f1_macro",
            problem_type="multiclass", save_path=None, bins=None, verbose="dont_show",
            random_state=42, n_classes=self.n_classes
        )
        xgb_multiclass.fit_and_get_model_and_results()
        xgb_multiclass.execute_shap_analysis(sample=True, num_features_to_show=10, num_sample=100)

    def _cat_boost_multiclass_classifier(self):
        cat_hyperparams = HxCatBoostHyperparameterBuilder().get_random_hyperparams()
        cat_multiclass = HxCatBoostMulticlassClassifier(
            data_dict=self.data_dict, hiperparams=cat_hyperparams, metric="f1_macro",
            problem_type="multiclass", save_path=None, bins=None, verbose="dont_show",
            random_state=42, n_classes=self.n_classes
        )
        cat_multiclass.fit_and_get_model_and_results()
        # cat_multiclass.execute_shap_analysis(sample=True, num_features_to_show=10, num_sample=100)

    def _dnn_multiclass_classifier(self):
        dnn_hyperparams = HxDnnHyperparameterBuilder(
            {'units': 16, 'dropout': 0.5, 'learning_rate': 0.01, 'regularization': 0.1, 'epochs': 150}
        ).validate_and_get_hyperparams()
        dnn = HxDenseNeuralNetworkMulticlassClassifier(
            self.data_dict, dnn_hyperparams, 'accuracy', 'multiclass', 1, 1,
            "relu", "l1", "adam", True, True, 15, None, None, "dont_show", 42
        )
        dnn.fit_and_get_model_and_results()
        dnn.execute_shap_analysis(sample=True, num_features_to_show=100, num_sample=200, background_sample=100)
    # </editor-fold>


class DigitsMulticlassExample:
    def __init__(self):
        """
        Ejemplo de clasificación MULTICLASE con dataset de Dígitos escritos a mano (10 clases)
        """
        self.IT: InfoTools = InfoTools()
        self.IT.header_print(f"DigitsMulticlassExample: Ejemplo de clasificación MULTICLASE con sklearn.datasets.digits utilizando hx_ml_dl_tools")

        # --------------------------------------------------------------------------------------
        # -- 1: Dataset multiclase
        # --------------------------------------------------------------------------------------
        self.digits_dataset = load_digits()
        self.x = pd.DataFrame(self.digits_dataset.data, columns=[f"pixel_{i}" for i in range(self.digits_dataset.data.shape[1])])
        self.y = pd.DataFrame(self.digits_dataset.target, columns=["target"])

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        self.data_dict = {
            "x_train": self.x_train,
            "y_train": self.y_train,
            "x_test": self.x_test,
            "y_test": self.y_test
        }
        self.n_classes = len(np.unique(self.y))

        # --------------------------------------------------------------------------------------
        # -- 2: Info dataset
        # --------------------------------------------------------------------------------------
        self.IT.sub_intro_print(f"Dataset Digits (Multiclase) cargado:")
        self.IT.info_print(f"Muestras totales: {self.x.shape[0]}")
        self.IT.info_print(f"Características: {self.x.shape[1]}")
        self.IT.info_print(f"Número de clases: {self.n_classes}")
        self.IT.info_print(f"Train samples: {self.x_train.shape[0]}")
        self.IT.info_print(f"Test samples: {self.x_test.shape[0]}")

        self.get_dataset_info()

    def execute_example(self):
        self._light_gbm_multiclass_classifier()
        self._xg_boost_multiclass_classifier()
        self._cat_boost_multiclass_classifier()
        self._dnn_multiclass_classifier()

    def get_dataset_info(self):
        self.IT.sub_intro_print("Distribución por clase (dígitos):")
        class_counts = self.y['target'].value_counts().sort_index()
        for digit, count in class_counts.items():
            self.IT.info_print(f"  - Dígito {digit}: {count} muestras ({(count/len(self.y))*100:.1f}%)")

    # <editor-fold desc="Modelos Multiclase ------------------------------------------------------------------------------------------------------------------------------------->

    def _light_gbm_multiclass_classifier(self):
        lgbm_hyperparams = HxLightgbmHyperparameterBuilder().get_random_hyperparams()
        lgbm_multiclass = HxLightGbmMulticlassClassifier(
            data_dict=self.data_dict, hiperparams=lgbm_hyperparams, metric="f1_macro",
            problem_type="multiclass", save_path=None, bins=None, verbose="dont_show",
            random_state=42, n_classes=self.n_classes
        )
        lgbm_multiclass.fit_and_get_model_and_results()
        lgbm_multiclass.execute_shap_analysis(sample=True, num_features_to_show=10, num_sample=100)

    def _xg_boost_multiclass_classifier(self):
        xgb_hyperparams = HxXgboostHyperparameterBuilder().get_random_hyperparams()
        xgb_multiclass = HxXtremeGradientBoostingMulticlassClassifier(
            data_dict=self.data_dict, hiperparams=xgb_hyperparams, metric="f1_macro",
            problem_type="multiclass", save_path=None, bins=None, verbose="dont_show",
            random_state=42, n_classes=self.n_classes
        )
        xgb_multiclass.fit_and_get_model_and_results()
        xgb_multiclass.execute_shap_analysis(sample=True, num_features_to_show=10, num_sample=100)

    def _cat_boost_multiclass_classifier(self):
        cat_hyperparams = HxCatBoostHyperparameterBuilder().get_random_hyperparams()
        cat_multiclass = HxCatBoostMulticlassClassifier(
            data_dict=self.data_dict, hiperparams=cat_hyperparams, metric="f1_macro",
            problem_type="multiclass", save_path=None, bins=None, verbose="dont_show",
            random_state=42, n_classes=self.n_classes
        )
        cat_multiclass.fit_and_get_model_and_results()
        # cat_multiclass.execute_shap_analysis(sample=True, num_features_to_show=10, num_sample=100)

    def _dnn_multiclass_classifier(self):
        dnn_hyperparams = HxDnnHyperparameterBuilder(
            {'units': 32, 'dropout': 0.5, 'learning_rate': 0.01, 'regularization': 0.1, 'epochs': 100}
        ).validate_and_get_hyperparams()
        dnn = HxDenseNeuralNetworkMulticlassClassifier(
            self.data_dict, dnn_hyperparams, 'accuracy', 'multiclass', 1, 2,
            "relu", "l2", "adam", True, True, 15, None, None, "dont_show", 42
        )
        dnn.fit_and_get_model_and_results()
        dnn.execute_shap_analysis(sample=True, num_features_to_show=100, num_sample=200, background_sample=100)
    # </editor-fold>
