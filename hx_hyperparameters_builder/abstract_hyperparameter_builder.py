from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import random


class AbstractHyperparameterBuilder(ABC):
    def __init__(self,
                 bounds_range: Dict[str, Tuple[float, float]],
                 float_bound_list: List[str],
                 malformation_ranges_dict: Dict[str, Tuple[float, float]]):
        """
        Clase abstracta que va a permitir validar y generar grupos de hiperparametros
        :param bounds_range: Diccionario con tuplas (min, max) para cada hiperparámetro
        :param float_bound_list: Lista de hiperparámetros que son float
        :param malformation_ranges_dict: Diccionario con rangos para evaluar malformaciones
        """
        # -------------------------------------------------------------------------------
        # -- 1: Almaceno en propiedades
        # -------------------------------------------------------------------------------

        # ---- 1.1: Almaceno los bounds (tupla con valores razonables para los hiperparametros)
        self.bounds_range: Dict[str, Tuple[float, float]] = bounds_range

        # ---- 1.2: Almaceno en una lista los nombres de los bounds que son flotantes
        self.float_bound_list: List[str] = float_bound_list

        # ---- 1.3: Almaceno los rangos que voy a considerar de malformación (uso genético)
        self.malformation_ranges_dict: Dict[str, Tuple[float, float]] = malformation_ranges_dict

    def get_random_hyperparams(self) -> Dict[str, float]:
        """
        Metodo que genera un diccionario de hiperparámetros aleatorios
        respetando el tipo (float o int) y los bounds definidos.
        :return: Diccionario {nombre_parametro: valor_aleatorio}
        """

        # -- 1: Defino el diccionario de hiperparamentrs
        hyperparams = {}

        # -- 2: Itero y voy generando hiperparametros random teniendo en cuenta si es flotante o no
        for param, (low, high) in self.bounds_range.items():
            if param in self.float_bound_list:
                # Genera un float dentro del rango
                hyperparams[param] = random.uniform(low, high)
            else:
                # Genera un int dentro del rango
                hyperparams[param] = random.randint(int(low), int(high))
        return hyperparams

    @abstractmethod
    def validate_and_get_hyperparams(self):
        pass

    @abstractmethod
    def validate_malformation_ranges(self):
        pass

