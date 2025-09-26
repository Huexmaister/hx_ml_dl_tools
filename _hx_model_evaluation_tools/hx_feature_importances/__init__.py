from typing import List, Set, Tuple, Dict, Optional
from constants_and_tools import ConstantsAndTools
import pandas as pd
import numpy as np


class HxFeatureImportancesandCorrelations:
    """
    Clase para identificar y eliminar variables altamente correlacionadas basándose en su importancia/peso.

    Esta clase analiza la matriz de correlación de un conjunto de datos y elimina las variables
    que están altamente correlacionadas entre sí, manteniendo únicamente la variable con mayor
    peso/importancia dentro de cada grupo correlacionado.
    """

    def __init__(self, x_test: pd.DataFrame, weights: List[float], corr_limit: float = 0.97) -> None:
        """
        Inicializa la clase con los datos y parámetros necesarios.

        Args:
            x_test (pd.DataFrame): DataFrame con las variables a analizar
            weights (List[float]): Lista con los pesos/importancia de cada variable,
                                 debe tener el mismo orden que las columnas de x_test
            corr_limit (float, optional): Umbral de correlación por encima del cual
                                        se consideran variables altamente correlacionadas.
                                        Por defecto 0.97

        Raises:
            ValueError: Si el número de pesos no coincide con el número de columnas
        """
        # Inicializar herramientas y constantes
        self.CT: ConstantsAndTools = ConstantsAndTools()

        # Validar que el número de pesos coincida con el número de columnas
        if len(weights) != len(x_test.columns):
            raise ValueError(f"El número de pesos ({len(weights)}) debe coincidir "
                             f"con el número de columnas ({len(x_test.columns)})")

        # Almacenar datos de entrada
        self.X_test: pd.DataFrame = x_test
        self.weights: List[float] = weights
        self.corr_limit: float = corr_limit

        # Calcular matriz de correlación absoluta
        self.corr_matrix: pd.DataFrame = self.X_test.corr().abs()

    def _find_high_correlation_pairs(self) -> List[Tuple[int, int]]:
        """
        Encuentra todos los pares de variables con correlación superior al límite establecido.

        Solo considera la matriz triangular superior para evitar duplicados y la diagonal.

        Returns:
            List[Tuple[int, int]]: Lista de tuplas con los índices de las columnas
                                  altamente correlacionadas
        """
        # Crear máscara para la matriz triangular superior (excluyendo diagonal)
        upper_triangle_mask = np.triu(np.ones(self.corr_matrix.shape), k=1).astype(bool)

        # Aplicar máscara a la matriz de correlación
        upper_triangle = self.corr_matrix.where(upper_triangle_mask)

        # Encontrar pares con correlación superior al límite
        high_corr_indices = np.where(upper_triangle > self.corr_limit)
        high_corr_pairs = list(zip(high_corr_indices[0], high_corr_indices[1]))

        return high_corr_pairs

    def _group_correlated_variables(self, high_corr_pairs: List[Tuple[int, int]]) -> List[Set[str]]:
        """
        Agrupa variables altamente correlacionadas en conjuntos.

        Si dos variables están correlacionadas con una tercera, todas forman un grupo.

        Args:
            high_corr_pairs (List[Tuple[int, int]]): Pares de índices de variables correlacionadas

        Returns:
            List[Set[str]]: Lista de conjuntos, cada uno conteniendo nombres de variables
                           que están altamente correlacionadas entre sí
        """
        high_corr_groups: List[Set[str]] = []

        for i, j in high_corr_pairs:
            # Convertir índices a nombres de columnas
            col1 = self.corr_matrix.columns[i]
            col2 = self.corr_matrix.columns[j]

            # Buscar si alguna de las columnas ya pertenece a un grupo existente
            found_group: Optional[Set[str]] = None
            for group in high_corr_groups:
                if col1 in group or col2 in group:
                    found_group = group
                    break

            if found_group is not None:
                # Agregar ambas columnas al grupo existente
                found_group.update([col1, col2])
            else:
                # Crear nuevo grupo con ambas columnas
                high_corr_groups.append({col1, col2})

        return high_corr_groups

    def _identify_variables_to_drop(self, high_corr_groups: List[Set[str]]) -> List[str]:
        """
        Identifica qué variables eliminar de cada grupo correlacionado.

        En cada grupo, mantiene la variable con mayor peso y marca las demás para eliminación.

        Args:
            high_corr_groups (List[Set[str]]): Grupos de variables correlacionadas

        Returns:
            List[str]: Lista de nombres de columnas a eliminar
        """
        to_drop: List[str] = []

        for group in high_corr_groups:
            # Calcular importancia/peso de cada variable en el grupo
            importances: Dict[str, float] = {
                col: self.weights[self.X_test.columns.get_loc(col)]
                for col in group
            }

            # Encontrar la variable con mayor importancia
            max_importance_col = max(importances, key=importances.get)

            # Agregar todas las demás variables del grupo a la lista de eliminación
            variables_to_remove = [col for col in group if col != max_importance_col]
            to_drop.extend(variables_to_remove)

        return to_drop

    def _print_results(self, to_drop: List[str], x_filtered: pd.DataFrame, high_corr_groups: List[Set[str]]) -> None:
        """
        Imprime los resultados del análisis de correlaciones.

        Args:
            to_drop (List[str]): Variables eliminadas
            x_filtered (pd.DataFrame): DataFrame filtrado
            high_corr_groups (List[Set[str]]): Grupos de variables correlacionadas
        """
        # Imprimir variables eliminadas
        self.CT.IT.sub_intro_print("CorrelationsAndWeights 1: Variables a eliminar por estar muy correlacionadas y tener menos peso:")
        self.CT.IT.info_print(f"{to_drop}")

        # Imprimir variables restantes
        self.CT.IT.sub_intro_print("CorrelationsAndWeights 2: Variables restantes ordenadas por peso")
        remaining_variables = list(x_filtered.columns)
        self.CT.IT.info_print(f"{remaining_variables}")

        # Imprimir grupos de variables correlacionadas
        self.CT.IT.sub_intro_print("CorrelationsAndWeights 3: Grupos de variables altamente correlacionadas entre sí")

        if len(high_corr_groups) == 0:
            self.CT.IT.info_print(f"No hay variables con alta correlacion")
        else:
            for i, group in enumerate(high_corr_groups, 1):
                self.CT.IT.info_print(f"Grupo {i}: {group}")

    def run(self) -> pd.DataFrame:
        """
        Ejecuta el proceso completo de eliminación de variables correlacionadas.

        Proceso:
        1. Encuentra pares de variables altamente correlacionadas
        2. Agrupa las variables correlacionadas
        3. Identifica cuáles eliminar basándose en sus pesos
        4. Filtra el DataFrame eliminando las variables seleccionadas
        5. Imprime resultados del análisis

        Returns:
            pd.DataFrame: DataFrame filtrado sin las variables altamente correlacionadas
                         de menor importancia
        """
        # Paso 1: Encontrar pares de variables altamente correlacionadas
        high_corr_pairs = self._find_high_correlation_pairs()

        # Paso 2: Agrupar variables correlacionadas
        high_corr_groups = self._group_correlated_variables(high_corr_pairs)

        # Paso 3: Identificar variables a eliminar
        to_drop = self._identify_variables_to_drop(high_corr_groups)

        # Paso 4: Filtrar DataFrame eliminando columnas seleccionadas
        x_filtered = self.X_test.drop(columns=to_drop)

        # Paso 5: Imprimir resultados
        self._print_results(to_drop, x_filtered, high_corr_groups)

        return x_filtered

    def get_correlation_summary(self) -> Dict[str, any]:
        """
        Obtiene un resumen del análisis de correlaciones sin modificar los datos.

        Returns:
            Dict[str, any]: Diccionario con información resumida del análisis:
                - 'high_corr_pairs': Número de pares altamente correlacionados
                - 'correlation_groups': Número de grupos de variables correlacionadas
                - 'variables_to_drop': Número de variables que serían eliminadas
                - 'remaining_variables': Número de variables que permanecerían
        """
        high_corr_pairs = self._find_high_correlation_pairs()
        high_corr_groups = self._group_correlated_variables(high_corr_pairs)
        to_drop = self._identify_variables_to_drop(high_corr_groups)

        return {
            'high_corr_pairs': len(high_corr_pairs),
            'correlation_groups': len(high_corr_groups),
            'variables_to_drop': len(to_drop),
            'remaining_variables': len(self.X_test.columns) - len(to_drop),
            'correlation_threshold': self.corr_limit
        }
