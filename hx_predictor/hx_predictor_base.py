import numpy as np
from constants_and_tools import ConstantsAndTools
from typing import List, Literal, Any, Dict
from abc import ABC, abstractmethod
import pandas as pd


class HxPredictor(ABC):
    def __init__(self, df_to_predict: pd.DataFrame, model_directory: str, model_type: Literal['classifier', 'regressor'], verbose: bool = False):
        """
        Padre de las clases que se van a encargar de realizar predicciones tanto de machine learning como de deep learning.
        La idea es que a través del contenido de la carpeta que se crea con las instancias Hx de crear modelos, tengamos lo necesario
        para optimizar el proceso de predicción y tener transparenci en la información. Básicamente necesitamos 2 cosas del directorio:
        Por ejemplo, suponiendo un directorio 'LGBM_binary_classifier', en su contenido hay dos archivos que necesitamos:
        1: HxLightGbmClassifier.joblib --> Modelo
        2: complete_train_result.json --> Diccionario con columnas de predicción, métricas y resumen del entrenamiento
        :param df_to_predict: Dataframe imputado y escalado sobre el que tenemos que predecir
        :param model_directory: Carpeta donde está el modelo y el diccionario de entrenamiento
        :param model_type: Tipo de modelo (lo necesito para el predict)
        :param verbose: True para mostrar información del modelo y del proceso, False para que no muestre nada
        """

        self.CT: ConstantsAndTools = ConstantsAndTools()

        # -----------------------------------------------------------------------------------------
        # -- 1: Almaceno los parámetros en propiedades de clase y cargo modelo y diccionario
        # -----------------------------------------------------------------------------------------

        # ---- 1.1: Almaceno los parámetros
        self.df_to_predict: pd.DataFrame = df_to_predict.copy()
        self.model_directory: str = model_directory
        self.verbose: bool = verbose
        self.model_type: Literal['classifier', 'regressor'] = model_type

        # ---- 1.2: Obtengo el json del diccionario
        self._json_files_list: List[str] = self.validate_just_one_file(self.CT.OT.get_path_files_by_extension(self.model_directory, extension=".json"), ".json")

        # ---- 1.3: Almaceno el diccionario
        self._train_dict: Dict = [z for z in self.CT.PdT.load_json_in_dict(self._json_files_list[0]).values()][0]

        # -----------------------------------------------------------------------------------------
        # -- 2: Almaceno la informacion del diccionario en propiedades
        # -----------------------------------------------------------------------------------------

        # ---- 2.1: Columnas a predecir
        self._pred_cols: List[str] = self._train_dict["columns"]

        # ---- 2.2: Columna target
        self._target_col: str = self._train_dict["target"]

        # ---- 2.3: Hyperparams
        self._hyperparams: Dict[str, Any] = self._train_dict['best_params']

        # ---- 2.5: TODO: Metrics, modificar el diccionario para ademas de metrics, tener train metrics y test metrics
        self._test_metrics: Dict[str, Any] = self._train_dict['metrics']

        # -----------------------------------------------------------------------------------------
        # -- 3: Valido si las columnas de predicción están todas en el df que se ha pasado
        # -----------------------------------------------------------------------------------------

        # ---- 3.1: Ejecuto el metodo
        self.validate_prediction_cols_in_df()

    @staticmethod
    def validate_just_one_file(files_list: List[str], extension: str) -> List[str]:
        """
        Metodo que valida si en una lista hay un solo archivo y en caso contrario lanza excepcion
        :param extension:
        :param files_list:
        :return:
        """

        # -- COmpruebo el len de la lista, si es distinto de 1, lanzo excepciones
        if len(files_list) > 1:
            raise ValueError(f"HxPredictor Error: En el directorio que se ha facilitado, hay mas de un archivo del siguiente tipo: {extension}")
        elif len(files_list) < 1:
            raise ValueError(f"HxPredictor Error: En el directorio que se ha facilitado, no hay ficheros del siguiente tipo: {extension}")

        # -- Defino el caso correcto
        return files_list

    def validate_prediction_cols_in_df(self) -> None:
        """
        Este metodo va a validar si las columnas de prediccion existen
        :return:
        """

        # ---- Obtengo la lista de columnas con las que se entrenó el modelo y las comparo con las que trae el df
        pred_cols_not_in_df: List[str] = [z for z in self._pred_cols if z not in [x for x in self.df_to_predict.columns]]

        # ---- En caso de que alguna no esté, las pinto y lanzo excepcion
        if len(pred_cols_not_in_df) > 0:
            raise IndexError(f"Hay columnas con las que entrenó el modelo que no están en el df: {pred_cols_not_in_df}")

    def get_predictor_info(self) -> None:
        """
        Metodo que va a pintar las propiedades de interes del modelo
        :return:
        """

        self.CT.IT.sub_intro_print(f"Infomación del modelo")
        self.CT.IT.info_print(f"Target column: {self._target_col}")
        self.CT.IT.info_print(f"Hyperparameters: {self._hyperparams}")
        self.CT.IT.info_print(f"Test Metrics: {self._test_metrics}")

    @abstractmethod
    def predict(self) -> np.ndarray:
        pass