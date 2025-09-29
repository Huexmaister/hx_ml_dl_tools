from constants_and_tools import ConstantsAndTools
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Literal
import tensorflow as tf
from io import StringIO
import pandas as pd
import numpy as np
import contextlib
import datetime
import glob
import json
import os


class HxDeepLearningBaseModel(ABC):
    def __init__(self, data_dict: Dict[str, pd.DataFrame],
                 hiperparams: dict,
                 model_name: str,
                 geometry: Literal[0, 1, 2, 3] = 1,
                 hidden_layers: int = 2,
                 activation: Literal['relu', 'tanh', 'sigmoid'] = 'relu',
                 regularization_type: Literal['l1', 'l2'] = 'l2',
                 optimizer: Literal['sgd', 'adam', 'rmsprop'] = 'adam',
                 include_dropout: bool = True,
                 include_batch_norm: bool = True,
                 early_stopping_patience: int = 15,
                 save_path: str | None = None):
        """
        Clase abstracta para construir modelos de Deep Learning
        :param data_dict: Diccionario que contiene el x_train, y_train, x_test, y_test
        :param save_path: Path base sobre el que se van a almacenar los resultados
        :param hiperparams: Diccionario de hiperparametros
        :param model_name: Nombre del modelo (se asigna directamente en la clase correspondiente)
        :param geometry: Diseño de la red, por ejemplo si es '>' indica que cada capa va a tener mas neuronas que la anterior
        :param hidden_layers: Cantidad de capas que va a tener la red (sin contar con la de salida)
        """
        super().__init__()

        # --------------------------------------------------------------------------------------------
        # -- 0: Instancio los toolkits que necesito y defino el diccionario maestro de resultados
        # --------------------------------------------------------------------------------------------

        # ---- 0.1: Instancio constants and tools
        self.CT: ConstantsAndTools = ConstantsAndTools()

        # ---- 0.2: Diccionario de resultados
        self.master_result_dict: Dict[str, Any] = {}

        # ---- 0.3: Creo la lista de callbacks
        self.callbacks: List[tf.keras.callbacks.Callback] = []

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
        self.x_train: pd.DataFrame = data_dict["x_train"].copy()
        self.y_train: pd.DataFrame = data_dict["y_train"].copy().astype(np.int64)
        self.x_test: pd.DataFrame = data_dict["x_test"].copy()
        self.y_test: pd.DataFrame = data_dict["y_test"].copy().astype(np.int64)

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

        # ---- 2.5: Almaceno el geometry, y obtengo el geometry_symbol
        self.geometry: Literal[0, 1, 2, 3] = geometry
        self.geometry_dict: dict = {0: "<", 1: ">", 2: "<>", 3: "><"}
        self.geometry_symbol: str = self._get_geometry_symbol()

        # ---- 2.6: Almaceno el hidden_layers
        self.hidden_layers: int = hidden_layers

        # ---- 2.7: Almaceno la funcion de activacion
        self.activation: Literal['relu', 'tanh', 'sigmoid'] = activation

        # ---- 2.8: Almaceno el tipo de regularizacion
        self.regularization_type: Literal['l1', 'l2'] = regularization_type

        # ---- 2.8: Almaceno el optimizador
        self.optimizer: Literal['sgd', 'adam', 'rmsprop'] = optimizer

        # ---- 2.9: Almaceno el include_dropout
        self.include_dropout: bool = include_dropout

        # ---- 2.10: Almaceno el include_batch_norm
        self.include_batch_norm: bool = include_batch_norm

        # ---- 2.11: Almaceno el early_stopping_patience
        self.early_stopping_patience: int = early_stopping_patience



        # --------------------------------------------------------------------------------------------
        # -- 3: Me quedo con los DataFrames y creo datasets de TensorFlow
        # --------------------------------------------------------------------------------------------

        # ---- 3.1: Almaceno los DataFrames para referencia
        self.x_train_df: pd.DataFrame = self.x_train.copy()
        self.x_test_df: pd.DataFrame = self.x_test.copy()

        # ---- 3.2: Creo datasets de TensorFlow optimizados manteniendo los nombres de columnas

        # ------ 3.2.1: Transformo los df a arrays de numpy con float32
        x_train_array = self.x_train_df.values.astype('float32')
        x_test_array = self.x_test_df.values.astype('float32')
        y_train_array = self.y_train.values.astype('float32')
        y_test_array = self.y_test.values.astype('float32')

        # ------ 3.2.2: Creo los datasets con el batch size pertinente
        batch_size = self.hiperparams.get('batch_size', 32)

        # ------ 3.2.3: Obtengo los dataset en formato tensor para acelerar en entrenamiento
        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train_array, y_train_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self.test_dataset = tf.data.Dataset.from_tensor_slices((x_test_array, y_test_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)



    def _get_geometry_symbol(self) -> str:
        """
        Metodo que devuelve el simbolo del diccionario de geometría
        :return:
        """

        return self.geometry_dict[self.geometry]

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

    def _setup_training_callbacks(self) -> None:
        """
        Configura los callbacks para el entrenamiento

        Returns:
            Lista de callbacks
        """
        # Early Stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=self.verbose,
            min_delta=0.001
        )
        self.callbacks.append(early_stopping)

        # Reducción de Learning Rate
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.hiperparams.get('reduce_lr_patience', 8),
            min_lr=1e-7,
            verbose=self.verbose
        )
        self.callbacks.append(reduce_lr)

        # Model Checkpoint
        """checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_save_path, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        self.callbacks.append(checkpoint)"""

    @staticmethod
    def get_model_summary(model) -> List[Dict]:
        """
        Metodo que devuelve una lista de diccionarios con la arquitectura del modelo
        :param model:
        :return:
        """

        # --- Captura el summary como lista de strings ---
        stream = StringIO()
        with contextlib.redirect_stdout(stream):
            model.summary()
        summary_lines = stream.getvalue().splitlines()

        data = []
        for line in summary_lines:
            # Solo procesar líneas que tienen capas (las que empiezan con un espacio y '|')
            if line.startswith('│') or line.startswith('|') or line.startswith('\u2502'):
                parts = [p.strip() for p in line.split('│') if p.strip()]
                if len(parts) == 3 or len(parts) == 4:  # capas con parámetros
                    layer_name = parts[0]
                    output_shape = parts[1]
                    param_count = parts[-1].replace(',', '')
                    try:
                        param_count = int(param_count)
                    except:
                        param_count = 0
                    data.append({
                        "layer_name": layer_name,
                        "output_shape": output_shape,
                        "param_count": param_count
                    })

        df = pd.DataFrame(data)
        return df.to_dict(orient='records')

    def get_weights(self, model) -> Tuple[List, Dict]:
        """
        Obtiene los pesos/importancia de las características del modelo DNN

        Args:
            model: Modelo de Keras entrenado

        Returns:
            Lista con pesos de las características
        """
        try:
            # Para DNN, usar los pesos de la primera capa como importancia
            if hasattr(model, 'layers') and len(model.layers) > 0:
                first_layer = model.layers[0]
                if hasattr(first_layer, 'get_weights') and first_layer.get_weights():
                    weights = first_layer.get_weights()[0]  # Pesos de la capa
                    # Calcular importancia como valor absoluto promedio por característica
                    feature_importance = np.mean(np.abs(weights), axis=1)

                    # Crear diccionario de pesos por característica
                    model_weights_dict = {
                        col: float(weight) for col, weight in zip(
                            self.x_train_df.columns,
                            feature_importance if feature_importance is not None else [0] * len(self.x_train_df.columns)
                        )
                    }

                    return feature_importance, model_weights_dict

        except Exception as e:
            self.CT.IT.warning_print(f"No se pudieron obtener pesos del modelo: {e}")

        return [np.array([1.0] * len(self.x_train_df.columns))] , {} # Fallback

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
