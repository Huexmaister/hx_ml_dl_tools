from hx_predictor.hx_predictor_base import HxPredictor
import numpy as np
from tensorflow.keras.models import Model
from typing import List, Literal
import tensorflow as tf
import pandas as pd


class HxDeepLearningPredictor(HxPredictor):
    def __init__(self, df_to_predict: pd.DataFrame, model_directory: str, model_type: Literal['classifier', 'regressor'], verbose: bool = False):
        super().__init__(df_to_predict, model_directory, model_type, verbose)

        # ---- Obtengo los joblib
        self._files_list: List[str] = self.validate_just_one_file(self.CT.OT.get_path_files_by_extension(self.model_directory, extension=".keras"), ".keras")

        # ---- Obtengo el path y cargo el modelo
        self._model_path: str = self._files_list[0]
        self._model = self.load_tensorflow_model()

    def load_tensorflow_model(self) -> Model | None:
        """
        Metodo para cargar un modelo de TensorFlow en formato .keras

        :return: Instancia del modelo cargado o None si falla la carga
        """

        # -----------------------------------------------------------------------------------------
        # -- 1: Intento cargar el modelo dentro de un bloque try-except
        # -----------------------------------------------------------------------------------------
        try:
            # ---- 1.1: Pinto la carga
            self.CT.IT.sub_intro_print(f"Cargando modelo TensorFlow desde {self._model_path} ...")

            # ---- 1.2: Cargo el modelo
            model: Model = tf.keras.models.load_model(self._model_path)

            # ---- 1.3: Pinto la información del modelo cargado
            self.CT.IT.info_print("Modelo cargado correctamente. Es una instancia de keras.Model", "green")
            self.CT.IT.info_print("Resumen del modelo:", "blue")

            # ---- 1.4: En caso de especificarlo con el verbose, pinto info del modelo
            if self.verbose:
                self.get_predictor_info()
                model.summary()

            # ---- 1.4: Retorno
            return model

        # -----------------------------------------------------------------------------------------
        # -- 2: Manejo de excepciones
        # -----------------------------------------------------------------------------------------
        except FileNotFoundError:
            self.CT.IT.info_print("Error al cargar el modelo, el path no existe", "light_red")
        except Exception as e:
            self.CT.IT.info_print(f"Error al cargar el modelo: {e}", "light_red")

    def predict(self) -> np.ndarray:
        """
        Metodo para realizar la prediccion.
        :return:
        """

        # -----------------------------------------------------------------------------------------
        # -- 1: Realizo y retorno la prediccion
        # -----------------------------------------------------------------------------------------

        # ---- 1.1: Realizo y retorno
        return self._model.predict(self.df_to_predict[self._pred_cols], verbose=0)