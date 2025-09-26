from constants_and_tools import ConstantsAndTools
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import datetime
import glob
import json
import os


class HxMachineLearningBaseModel(ABC):
    def __init__(self, data_dict: Dict[str, pd.DataFrame], hiperparams: dict, model_name: str, save_path: str | None = None):
        """
        Clase abstracta para construir modelos de Machine Learning
        :param data_dict: Diccionario que contiene el x_train, y_train, x_test, y_test
        :param save_path: Path base sobre el que se van a almacenar los resultados
        :param hiperparams: Diccionario de hiperparametros
        :param model_name: Nombre del modelo (se asigna directamente en la clase correspondiente)
        """
        super().__init__()

        # --------------------------------------------------------------------------------------------
        # -- 0: Instancio los toolkits que necesito y defino el diccionario maestro de resultados
        # --------------------------------------------------------------------------------------------

        # ---- 0.1: Instancio constants and tools
        self.CT: ConstantsAndTools = ConstantsAndTools()

        # ---- 0.2: Diccionario de resultados
        self.master_result_dict: Dict[str, Any] = {}

        # --------------------------------------------------------------------------------------------
        # -- 1: Obtenicion del día en el formato necesario y creación dinámica de paths
        # --------------------------------------------------------------------------------------------

        # ---- 1.1: Almaceno el día de hoy en la propidad
        self.today_str_daye: str = datetime.datetime.now().strftime('%Y_%m_%d')

        # ---- 1.2: Configuro el conjunto de paths en el que vamos a almacenar la info y los modelos

        # -------- 1.2.1: Si el path ha sido proporcionado y no existe, lo creo
        if save_path is not None:
            self.CT.OT.create_folder_if_not_exists(save_path)  # -- Creo el path base si no existe ya
            self.save_path: str = f"{save_path}/training_results"
        else:
            self.save_path: str = f"{self.CT.output_path}/training_results"

        # -------- 1.2.2: Creo el save_path base si no existe
        self.CT.OT.create_folder_if_not_exists(self.save_path)

        # -------- 1.2.3: Agrego el día al path y si no existe lo creo
        self.save_path: str = f"{self.save_path}/{self.today_str_daye}"
        self.CT.OT.create_folder_if_not_exists(self.save_path)

        # -------- 1.2.4: Listo las versiones (los entrenamientos) que se han creado en el día
        self.day_train_versions: List[str] = self.list_versions()

        # -------- 1.2.5: Obtengo la version actual (la necesitaré para validar si existe el modelo concreto en la versión y así saber si hay que crear otra)
        self.current_train_version: str = self.day_train_versions[-1]

        # -------- 1.2.6: Creo la version si no existe
        self.CT.OT.create_folder_if_not_exists(f"{self.save_path}/{self.current_train_version}")

        # --------------------------------------------------------------------------------------------
        # -- 2: Almaceno los parametros del constructor en propiedades y creo paths
        # --------------------------------------------------------------------------------------------

        # ---- 2.1: Almaceno los datasets que se van a usar para entrenar en propiedades
        self.x_train = data_dict["x_train"].copy()
        self.y_train = data_dict["y_train"].copy().astype(np.int64)
        self.x_test = data_dict["x_test"].copy()
        self.y_test = data_dict["y_test"].copy().astype(np.int64)

        # ---- 2.2: Almaceno los hiperparámetros en la propiedad
        self.hiperparams: dict = hiperparams

        # ---- 2.3: Almaceno las columna target en la propiedad y valido
        self.target_col_list: list = [col for col in self.y_train.columns]

        # -------- 2.3.1: Lanzo excepcion si no hay una columna
        if len(self.target_col_list) != 1:
            raise IndexError(f"El y_train contiene {len(self.target_col_list)} columnas objetivo. Debe contener una")

        # -------- 2.3.2: Si no ha lanzado excepcion, se asigna la target_col_name
        self.target_col_name: str = self.target_col_list[0]

        # ---- 2.4: Almaceno el model_name
        self.model_name: str = model_name

        # --------------------------------------------------------------------------------------------
        # -- 3: Me quedo con el x_test_df y el resto los transformo a ndarray
        # --------------------------------------------------------------------------------------------

        # ---- 3.1: Almaceno en x_test_df el dataframe de test y en x_train_df el df de train
        self.x_train_df: pd.DataFrame = self.x_train.copy()
        self.x_test_df: pd.DataFrame = self.x_test.copy()

        # ---- 3.2: Transformo a arrays de numpy para optimizar el entrenamiento
        self.x_train: np.ndarray = self.x_train.values
        self.y_train: np.ndarray = self.y_train.values
        self.x_test: np.ndarray = self.x_test.values
        self.y_test: np.ndarray = self.y_test.values

    def list_versions(self) -> List[str]:
        """
        Lista las versiones de los modelos entrenados en el día actual.

        Se usa para obtener un historial de entrenamientos diarios y generar la siguiente versión.
        En caso de que el historial esté vacío, se devuelve directamente una nueva version

        :return: Lista de nombres de carpetas que representan versiones existentes (p. ej. ["v_001", "v_002"]).
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Utilizo glob para obtener todas las carpetas que hay en el self.save_path
        # -------------------------------------------------------------------------------------------------

        # ---- 1.1: Obtengo las carpetas con glob
        all_folders = glob.glob(f"{self.save_path}/v_*")

        # ---- 1.2: Obtengo los path omitiendo caracteres de barra tipo \\
        version_list: List[str] = [os.path.basename(folder) for folder in all_folders if os.path.basename(folder).startswith("v_")]

        # -------------------------------------------------------------------------------------------------
        # -- 2: Ordeno la lista de versiones para que v_001 venga antes que v_002, etc.
        # -------------------------------------------------------------------------------------------------
        version_list.sort()

        # -------------------------------------------------------------------------------------------------
        # -- 3: En caso de que no existan versiones, devuelvo la lista con la primera version
        # -------------------------------------------------------------------------------------------------
        if len(version_list) == 0:
            version_list.append('v_001')

        return version_list

    def new_training_version(self) -> str:
        """
        Obtiene la última versión de entrenamiento del día y devuelve la siguiente versión disponible.
        Se usa para: crear un identificador de versión único para cada nuevo entrenamiento diario.

        :return: String con la nueva versión (p. ej. "v_001", "v_002", "v_003").
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Obtengo la lista de versiones del día actual
        # -------------------------------------------------------------------------------------------------
        versions: List = self.list_versions()

        # -------------------------------------------------------------------------------------------------
        # -- 2: Determino la nueva versión
        # -------------------------------------------------------------------------------------------------
        if not versions:  # ---- 2.1: Si no hay versiones previas, empezamos con v_001
            new_version = "v_001"
        else:  # ---- 2.2: Si existen versiones previas, incremento el número de la última
            # ---- 2.2.1: Tomo la última versión de la lista ordenada
            last_version = versions[-1]
            last_number = int(last_version.split("_")[1])
            # ---- 2.2.2: Incremento y formateo con 3 dígitos
            new_version = f"v_{last_number + 1:03d}"

        return new_version

    def save_metrics_dict_json(self, metrics_dict: Dict[str, Any], save_path: str, json_name: str = "metrics_result") -> None:
        """
        Metodo que guarda el diccionario de metricas en un json
        :param metrics_dict:
        :param save_path:
        :param json_name:
        :return:
        """
        # -- 0: Pinto la entrada
        self.CT.IT.sub_intro_print(f"Almacenando diccionario de metricas : {json_name} en el path: {save_path}...")

        # -- 1: Creo la carpeta si no existe
        os.makedirs(save_path, exist_ok=True)

        # -- 2: Almaceno el diccionario
        file_path = os.path.join(save_path, f'{json_name}.json')
        with open(file_path, 'w') as archivo_json:
            json.dump(metrics_dict, archivo_json, indent=4)

        # -- 3: Pinto la salida
        self.CT.IT.info_print(f"{json_name} se ha guardado correctamente en {save_path}")

    # <editor-fold desc="Propiedades abstractas   --------------------------------------------------------------------------------------------------------------------------------">

    # -- Defino la propiedad verbose
    @property
    @abstractmethod
    def verbose(self) -> int | bool:
        """
        Implementacion del verbose de los modelos, cada uno tiene sus propios valores para:
        - Mostrar completo
        - Mostrar resumen
        - No mostrar
        """
        pass

    # -- Defino la propiedad metric
    @property
    @abstractmethod
    def metric(self) -> str:
        """
        Implementacion de la metrica de los modelos (los regresores tendran un literal concreto y los clasificadorees otro)

        """
        pass

    # -- Defino la propiedad problem_type
    @property
    @abstractmethod
    def problem_type(self) -> str:
        """
        Implementacion del tipo de problema que se va a abordar (binary, multiclass, regression, forecasting)

        """
        pass

    # -- Defino el setter de la propiedad problem_type
    @problem_type.setter
    def problem_type(self, value):
        self.problem_type = value


    # </editor-fold>

    # <editor-fold desc="Metodos abstractos   ------------------------------------------------------------------------------------------------------------------------------------">

    # -- Defino el metodo get_weights
    @abstractmethod
    def get_save_path(self) -> str:
        """
        Metodo para obtener (o crear y obtener) el path donde se van a almacenar los resultados

        """
        pass

    # -- Defino el metodo get_weights
    @abstractmethod
    def get_weights(self, model) -> List:
        """
        Implementacion del metodo que devuelve los pesos del modelo

        """
        pass

    # -- Defino el metodo fit_and_get_metrics
    @abstractmethod
    def fit_and_get_metrics(self, save_metrics_dict: bool) -> Dict[str, Any]:
        """
        Implementacion del metodo que entrena y devuelve el diccionario de métricas

        """
        pass

    # -- Defino el metodo fit_and_get_metrics
    @abstractmethod
    def fit_and_get_model_and_results(self) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any], List]:
        """
        Implementacion del metodo que entrena y devuelve model, x_test_probs, x_train_probs, self.master_dict, model_weights

        """
        pass

    # </editor-fold>
