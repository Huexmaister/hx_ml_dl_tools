import numpy as np
from hx_predictor.hx_predictor_base import HxPredictor
from typing import List, Literal
import pandas as pd
import joblib


class HxMachineLearningPredictor(HxPredictor):
    def __init__(self, df_to_predict: pd.DataFrame, model_directory: str, model_type: Literal['classifier', 'regressor'], verbose: bool = False):
        super().__init__(df_to_predict, model_directory, model_type, verbose)

        # ---- Obtengo los joblib
        self._files_list: List[str] = self.validate_just_one_file(self.CT.OT.get_path_files_by_extension(self.model_directory, extension=".joblib"), ".joblib")

        # ---- Obtengo el path y cargo el modelo
        self._model_path: str = self._files_list[0]
        self._model = self.load_machine_learning_model()

    def load_machine_learning_model(self):
        """
        Metodo para cargar un modelo de machine learning
        :return:
        """

        # -----------------------------------------------------------------------------------------
        # -- 1: Cargo el modelo en un try except
        # -----------------------------------------------------------------------------------------

        try:

            # ---- 1.1: Pinto la carga
            self.CT.IT.sub_intro_print(f"Cargando modelo {self._model_path} ...")

            # ---- 1.2: Cargo el modelo
            model = joblib.load(self._model_path)

            # ---- 1.3: Pinto la informacion del modelo cargado
            self.CT.IT.info_print(f"Modelo cargado correctamente. Es un {type(model).__name__} del modulo {type(model).__module__}")

            # ---- 1.4: En caso de especificarlo con el verbose, pinto info del modelo
            if self.verbose:
                self.get_predictor_info()

            # ---- 1.5: Retorno
            return model

        # -----------------------------------------------------------------------------------------
        # -- 2: Defino las excepciones, una para el path no existe y otra genérica
        # -----------------------------------------------------------------------------------------

        except FileNotFoundError:
            self.CT.IT.info_print("Error al cargar el modelo, el path no existe", "light_red")
        except Exception as e:
            self.CT.IT.info_print(f"Error al cargar el modelo: {e}", "light_red")

    def predict(self) -> np.ndarray:
        """
        Metodo para realizar la prediccion. Diferencia entre clasificador y regresor.
        Clasificador: Retorna el predict proba
        Regresor: Retorna la prediccion estandar
        :return:
        """

        # -----------------------------------------------------------------------------------------
        # -- 1: En funcion del tipo de modelo, realizo la predicción o la predict_proba
        # -----------------------------------------------------------------------------------------

        # ---- 1.1: Clasificadores
        if self.model_type == "classifier":
            return self._model.predict_proba(self.df_to_predict[self._pred_cols])[:, 1]

        # ---- 1.2: Regresores
        else:
            return self._model.predict(self.df_to_predict[self._pred_cols])
