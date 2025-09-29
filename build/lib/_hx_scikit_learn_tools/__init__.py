from imblearn.under_sampling import TomekLinks, RandomUnderSampler, CondensedNearestNeighbour, NeighbourhoodCleaningRule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, OneHotEncoder
from category_encoders import TargetEncoder, CatBoostEncoder, JamesSteinEncoder, LeaveOneOutEncoder, SumEncoder
from imblearn.over_sampling import SMOTE
from singleton_tools import SingletonMeta
from typing import Tuple, Optional
import pandas as pd


class SkLearnTools(metaclass=SingletonMeta):
    def __init__(self):
        pass

    # <editor-fold desc="Split    ------------------------------------------------------------------------------------------------------------------------------------------------">

    @staticmethod
    def split_train_test_forecasting(df: pd.DataFrame, date_colname: str, limit_date_str: Optional[str] = None, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide un DataFrame en conjuntos de entrenamiento y prueba para forecasting.

        Existen dos modos de partición:
        1. Si se proporciona `limit_date_str`, se usará como fecha de corte.
           - Todas las filas con fecha <= límite irán al conjunto de entrenamiento.
           - Todas las filas con fecha > límite irán al conjunto de prueba.
        2. Si no se especifica `limit_date_str`, se dividirá el dataset
           automáticamente según el porcentaje definido en `test_size`.

        :param df: (pd.DataFrame): DataFrame de entrada que contiene los datos.
        :param date_colname: (str): Nombre de la columna que contiene las fechas.
        :param limit_date_str: (Optional[str]): Fecha límite en formato 'YYYY-MM-DD'. Si es None, se usará `test_size`.
        :param test_size: (float, optional): Proporción de datos para el conjunto de prueba (valor entre 0 y 1). Default = 0.2.
        :return: Tuple[pd.DataFrame, pd.DataFrame]
        """


        # Caso 1: Partición por fecha límite explícita
        if limit_date_str is not None:
            # Convertimos la fecha límite a tipo datetime
            limit_date = pd.to_datetime(limit_date_str, format="%Y-%m-%d")

            # Entrenamiento: filas hasta la fecha límite (incluida)
            df_train = df[df[date_colname] <= limit_date].copy()

            # Prueba: filas posteriores a la fecha límite
            df_test = df[df[date_colname] > limit_date].copy()

            return df_train, df_test

        # Caso 2: Partición automática por proporción
        # Ordenamos por columna de fecha para mantener la secuencia temporal
        df_sorted = df.sort_values(by=date_colname)

        # Calculamos tamaño del set de entrenamiento
        train_size = int((1 - test_size) * len(df_sorted))

        # División en train y test
        df_train = df_sorted.iloc[:train_size].copy()
        df_test = df_sorted.iloc[train_size:].copy()

        return df_train, df_test

    @staticmethod
    def split_train_test(df: pd.DataFrame, test_size: float = 0.2, shuffle: bool = False, random_state: Optional[int] = 42, stratify_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide un DataFrame en conjuntos de entrenamiento y prueba.

        Se puede estratificar por una columna específica para mantener
        la distribución de clases en train y test.

        :param df: DataFrame de entrada.
        :param test_size: Proporción de datos para el conjunto de prueba (entre 0 y 1).
        :param shuffle: Indica si se deben mezclar las filas antes de dividir.
        :param random_state: Semilla aleatoria para reproducibilidad.
        :param stratify_col: Nombre de la columna a usar para estratificación. Si es None, no se aplica estratificación.
        :return: Una tupla (df_train, df_test).
        """
        # -------------------------------------------------------------------------------------------------
        # -- 1: Evaluamos si existe columna de estratificacion
        # -------------------------------------------------------------------------------------------------

        # ---- 1.1: Si no existe, retornamos el train test_split con normalidad
        if stratify_col is None:
            return train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=random_state)

        # ---- 1.2: Si existe, la agregamos al parametro stratify y retornamos
        return train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=random_state, stratify=df[stratify_col])

    # </editor-fold>

    # <editor-fold desc="Forecasting    ------------------------------------------------------------------------------------------------------------------------------------------">

    # </editor-fold>

    # <editor-fold desc="Oversamplers    -----------------------------------------------------------------------------------------------------------------------------------------">
    @staticmethod
    def smote_oversampler(x_train: pd.DataFrame, y_train: pd.Series, sampling_prop: Optional[float] = 0.8) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aplica la técnica SMOTE (Synthetic Minority Over-sampling Technique) para balancear
        las clases en un conjunto de entrenamiento. Utiliza k-NN para generar ejemplos sintéticos
        de la clase minoritaria en lugar de eliminar ejemplos de la clase mayoritaria.

        :param x_train: Conjunto de características de entrenamiento.
        :param y_train: Conjunto de etiquetas de entrenamiento.
        :param sampling_prop: Proporción de muestreo para la clase minoritaria. Si es None, se usa el valor por defecto de SMOTE.
        :return: Una tupla (X_smote, y_smote) con los datos balanceados.
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Configuración del oversampler SMOTE
        # -------------------------------------------------------------------------------------------------
        smote: SMOTE = SMOTE(sampling_strategy=sampling_prop)

        # -------------------------------------------------------------------------------------------------
        # -- 2: Aplicación del metodo fit_resample para generar nuevos ejemplos
        # -------------------------------------------------------------------------------------------------
        x_smote, y_smote = smote.fit_resample(x_train, y_train)

        # -------------------------------------------------------------------------------------------------
        # -- 3: Retorno de los datos balanceados
        # -------------------------------------------------------------------------------------------------
        return x_smote, y_smote

    # </editor-fold>

    # <editor-fold desc="Undersamplers    ----------------------------------------------------------------------------------------------------------------------------------------">
    @staticmethod
    def neighbourhood_cleaning_rule_undersampler(x_train: pd.DataFrame, y_train: pd.Series, sampling_prop: Optional[float] = 0.6) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aplica Neighborhood Cleaning Rule (NCR) para eliminar ejemplos de la clase mayoritaria
        mal clasificados o rodeados por ejemplos de la clase minoritaria. Posteriormente,
        puede aplicar Random Under-Sampling para ajustar la proporción.

        :param x_train: Conjunto de características de entrenamiento.
        :param y_train: Conjunto de etiquetas de entrenamiento.
        :param sampling_prop: Proporción de muestreo para la clase mayoritaria. Si es None, se aplica únicamente NCR.
        :return: Una tupla (x_resampled, y_resampled) con los datos balanceados.
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Aplicación de Neighborhood Cleaning Rule (NCR)
        # -------------------------------------------------------------------------------------------------
        ncr: NeighbourhoodCleaningRule = NeighbourhoodCleaningRule()
        x_ncr, y_ncr = ncr.fit_resample(x_train, y_train)

        # -------------------------------------------------------------------------------------------------
        # -- 2: Si no se especifica sampling_prop, retornamos directamente NCR
        # -------------------------------------------------------------------------------------------------
        if sampling_prop is None:
            return x_ncr, y_ncr

        # -------------------------------------------------------------------------------------------------
        # -- 3: Aplicamos Random Under-Sampling posterior a NCR
        # -------------------------------------------------------------------------------------------------
        rus: RandomUnderSampler = RandomUnderSampler(sampling_strategy=sampling_prop)
        x_resampled, y_resampled = rus.fit_resample(x_ncr, y_ncr)

        return x_resampled, y_resampled

    @staticmethod
    def condensed_nearest_neighbour_undersampler(x_train: pd.DataFrame, y_train: pd.Series, sampling_prop: Optional[float] = 0.5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aplica Condensed Nearest Neighbour (CNN) para seleccionar un subconjunto mínimo
        de ejemplos necesarios para clasificar correctamente la clase minoritaria.
        Posteriormente, puede aplicar Random Under-Sampling para ajustar la proporción.

        :param x_train: Conjunto de características de entrenamiento.
        :param y_train: Conjunto de etiquetas de entrenamiento.
        :param sampling_prop: Proporción de muestreo para la clase mayoritaria. Si es None, se aplica únicamente CNN.
        :return: Una tupla (x_resampled, y_resampled) con los datos balanceados.
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Aplicación de Condensed Nearest Neighbour (CNN)
        # -------------------------------------------------------------------------------------------------
        cnn: CondensedNearestNeighbour = CondensedNearestNeighbour()
        x_cnn, y_cnn = cnn.fit_resample(x_train, y_train)

        # -------------------------------------------------------------------------------------------------
        # -- 2: Si no se especifica sampling_prop, retornamos directamente CNN
        # -------------------------------------------------------------------------------------------------
        if sampling_prop is None:
            return x_cnn, y_cnn

        # -------------------------------------------------------------------------------------------------
        # -- 3: Aplicamos Random Under-Sampling posterior a CNN
        # -------------------------------------------------------------------------------------------------
        rus: RandomUnderSampler = RandomUnderSampler(sampling_strategy=sampling_prop)
        x_resampled, y_resampled = rus.fit_resample(x_cnn, y_cnn)

        return x_resampled, y_resampled

    @staticmethod
    def tomek_links_undersampling(x_train: pd.DataFrame, y_train: pd.Series, sampling_prop: Optional[float] = 0.5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aplica Tomek Links, identificando pares de ejemplos de clases opuestas
        que son vecinos más cercanos entre sí. Los ejemplos de la clase mayoritaria
        en cada par se eliminan. Posteriormente, puede aplicar Random Under-Sampling.

        :param x_train: Conjunto de características de entrenamiento.
        :param y_train: Conjunto de etiquetas de entrenamiento.
        :param sampling_prop: Proporción de muestreo para la clase mayoritaria. Si es None, se aplica únicamente Tomek Links.
        :return: Una tupla (x_resampled, y_resampled) con los datos balanceados.
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Aplicación de Tomek Links
        # -------------------------------------------------------------------------------------------------
        tl: TomekLinks = TomekLinks()
        x_tl, y_tl = tl.fit_resample(x_train, y_train)

        # -------------------------------------------------------------------------------------------------
        # -- 2: Si no se especifica sampling_prop, retornamos directamente TL
        # -------------------------------------------------------------------------------------------------
        if sampling_prop is None:
            return x_tl, y_tl

        # -------------------------------------------------------------------------------------------------
        # -- 3: Aplicamos Random Under-Sampling posterior a TL
        # -------------------------------------------------------------------------------------------------
        rus: RandomUnderSampler = RandomUnderSampler(sampling_strategy=sampling_prop)
        x_resampled, y_resampled = rus.fit_resample(x_tl, y_tl)

        return x_resampled, y_resampled

    # </editor-fold>

    # <editor-fold desc="Encoders para aplicar al dataframe completo (sin separar en train y test)    ----------------------------------------------------------------------------">

    @staticmethod
    def ordinal_encoder(df_column: pd.Series, jerarquized_list: list[str]) -> pd.DataFrame:
        """
        Aplica codificación ordinal a una columna categórica jerarquizada.
        Asigna valores numéricos consecutivos a categorías que tienen un orden jerárquico.

        Ejemplo:
            primaria -> 0
            secundaria -> 1
            universidad -> 2

        :param df_column: Columna del DataFrame a codificar.
        :param jerarquized_list: Lista ORDENADA DE MENOR A MAYOR de las categorías.
        :return: Columna transformada como DataFrame con valores ordinales.
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Inicialización del codificador ordinal con la jerarquía definida
        # -------------------------------------------------------------------------------------------------
        ordinal_encoder: OrdinalEncoder = OrdinalEncoder(categories=[jerarquized_list])

        # -------------------------------------------------------------------------------------------------
        # -- 2: Transformación de la columna en un array 2D y aplicación del encoder
        # -------------------------------------------------------------------------------------------------
        encoded_column = ordinal_encoder.fit_transform(df_column.to_numpy().reshape(-1, 1))

        # -------------------------------------------------------------------------------------------------
        # -- 3: Retorno como DataFrame
        # -------------------------------------------------------------------------------------------------
        return pd.DataFrame(encoded_column, columns=[df_column.name])

    @staticmethod
    def one_hot_encoder(df: pd.DataFrame, cols_to_encode: list[str]) -> Tuple[pd.DataFrame, list[str]]:
        """
        Aplica One-Hot Encoding a las columnas seleccionadas de un DataFrame.
        Genera nuevas columnas binarias para cada categoría encontrada.

        :param df: DataFrame de entrada.
        :param cols_to_encode: Lista de columnas a codificar.
        :return: Una tupla (new_df, feature_names).
                 - new_df: DataFrame con las columnas codificadas.
                 - feature_names: Lista de los nombres de las columnas creadas.
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Inicialización del codificador OneHot con soporte para categorías desconocidas
        # -------------------------------------------------------------------------------------------------
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        # -------------------------------------------------------------------------------------------------
        # -- 2: Ajuste del encoder a las categorías presentes en las columnas seleccionadas
        # -------------------------------------------------------------------------------------------------
        encoder.fit(df[cols_to_encode])

        # -------------------------------------------------------------------------------------------------
        # -- 3: Transformación de las columnas categóricas
        # -------------------------------------------------------------------------------------------------
        encoded_cols = encoder.transform(df[cols_to_encode])

        # -------------------------------------------------------------------------------------------------
        # -- 4: Obtención de los nombres de las nuevas columnas
        # -------------------------------------------------------------------------------------------------
        feature_names = encoder.get_feature_names_out(cols_to_encode).tolist()

        # -------------------------------------------------------------------------------------------------
        # -- 5: Creación de un DataFrame con las columnas codificadas
        # -------------------------------------------------------------------------------------------------
        encoded_df = pd.DataFrame(encoded_cols, columns=feature_names, index=df.index)

        # -------------------------------------------------------------------------------------------------
        # -- 6: Concatenación del DataFrame original (sin las columnas originales) con las codificadas
        # -------------------------------------------------------------------------------------------------
        new_df = pd.concat([df.drop(cols_to_encode, axis=1), encoded_df], axis=1)

        return new_df, feature_names

    @staticmethod
    def one_hot_encoder_pd(df: pd.DataFrame, cols_to_encode: list[str]) -> pd.DataFrame:
        """
        Aplica One-Hot Encoding usando pandas.get_dummies a las columnas seleccionadas.
        Genera nuevas columnas binarias para cada categoría encontrada.

        :param df: DataFrame de entrada.
        :param cols_to_encode: Lista de columnas a codificar.
        :return: DataFrame con las columnas codificadas.
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Aplicación de get_dummies a las columnas seleccionadas
        # -------------------------------------------------------------------------------------------------
        encoded_df = pd.get_dummies(df, columns=cols_to_encode)

        return encoded_df

    # </editor-fold>

    # <editor-fold desc="Encoders para aplicar al conjunto de datos separado en train y test    ----------------------------------------------------------------------------------">

    @staticmethod
    def catboost_encoder(train_df: pd.DataFrame, test_df: pd.DataFrame, target: pd.Series, cols_to_encode: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Codifica variables categóricas utilizando CatBoostEncoder.
        Similar al Target Encoding pero más robusto frente a overfitting y outliers.

        ➤ Usar cuando: buscas un encoding potente y robusto para modelos de árbol
          y tienes suficiente cantidad de datos por categoría.

        :param train_df: DataFrame de entrenamiento (sin columna objetivo).
        :param test_df: DataFrame de prueba (sin columna objetivo).
        :param target: Serie objetivo de entrenamiento.
        :param cols_to_encode: Lista de columnas a codificar.
        :return: Tupla (train_df_transformado, test_df_transformado).
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Copias y validación básica de columnas
        # -------------------------------------------------------------------------------------------------
        train_df: pd.DataFrame = train_df.copy()
        test_df: pd.DataFrame = test_df.copy()
        present_cols = [c for c in cols_to_encode if c in train_df.columns and c in test_df.columns]

        # ---- 1.1: Si no hay columnas válidas, devolvemos copias sin tocar
        if not present_cols:
            return train_df, test_df

        # -------------------------------------------------------------------------------------------------
        # -- 2: Aseguramos tipo category en las columnas a codificar
        # -------------------------------------------------------------------------------------------------
        for col in present_cols:
            train_df[col] = train_df[col].astype("category")
            test_df[col] = test_df[col].astype("category")

        # -------------------------------------------------------------------------------------------------
        # -- 3: Fit y transform con CatBoostEncoder
        # -------------------------------------------------------------------------------------------------
        encoder: CatBoostEncoder = CatBoostEncoder()
        encoder.fit(train_df[present_cols], target)

        train_df[present_cols] = encoder.transform(train_df[present_cols])
        test_df[present_cols] = encoder.transform(test_df[present_cols])

        # -------------------------------------------------------------------------------------------------
        # -- 4: Retorno
        # -------------------------------------------------------------------------------------------------
        return train_df, test_df

    @staticmethod
    def james_stein_encoder(train_df: pd.DataFrame, test_df: pd.DataFrame, target: pd.Series, cols_to_encode: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Codifica variables categóricas usando James-Stein encoder.
        Reduce varianza combinando frecuencia y media del target.

        ➤ Usar cuando: quieres regularizar el encoding y mejorar generalización,
          especialmente en categorías con tamaños variados.

        :param train_df: DataFrame de entrenamiento (sin columna objetivo).
        :param test_df: DataFrame de prueba (sin columna objetivo).
        :param target: Serie objetivo de entrenamiento.
        :param cols_to_encode: Lista de columnas a codificar.
        :return: Tupla (train_df_transformado, test_df_transformado).
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Copias y validación básica de columnas
        # -------------------------------------------------------------------------------------------------
        train_df: pd.DataFrame = train_df.copy()
        test_df: pd.DataFrame = test_df.copy()
        present_cols = [c for c in cols_to_encode if c in train_df.columns and c in test_df.columns]
        if not present_cols:
            return train_df, test_df

        # -------------------------------------------------------------------------------------------------
        # -- 2: Aseguramos tipo category en las columnas a codificar
        # -------------------------------------------------------------------------------------------------
        for col in present_cols:
            train_df[col] = train_df[col].astype("category")
            test_df[col] = test_df[col].astype("category")

        # -------------------------------------------------------------------------------------------------
        # -- 3: Fit_transform en train y transform en test con JamesSteinEncoder
        # -------------------------------------------------------------------------------------------------
        encoder: JamesSteinEncoder = JamesSteinEncoder()
        train_encoded = encoder.fit_transform(train_df[present_cols], target)
        test_encoded = encoder.transform(test_df[present_cols])

        # -------------------------------------------------------------------------------------------------
        # -- 4: Reconstrucción de dataframes (sustituimos columnas originales por codificadas)
        # -------------------------------------------------------------------------------------------------
        train_df = pd.concat([train_df.drop(present_cols, axis=1), train_encoded], axis=1)
        test_df = pd.concat([test_df.drop(present_cols, axis=1), test_encoded], axis=1)

        return train_df, test_df

    @staticmethod
    def leave_one_out_encoder(train_df: pd.DataFrame, test_df: pd.DataFrame, target: pd.Series, cols_to_encode: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Codifica variables categóricas usando Leave-One-Out encoding.
        Calcula la media del target excluyendo la observación actual para evitar sesgo.

        ➤ Usar cuando: quieres un encoding que reduzca el sesgo del target respecto a la propia fila,
          pero ten cuidado con categorías con muy pocas observaciones (riesgo overfitting).

        :param train_df: DataFrame de entrenamiento (sin columna objetivo).
        :param test_df: DataFrame de prueba (sin columna objetivo).
        :param target: Serie objetivo de entrenamiento.
        :param cols_to_encode: Lista de columnas a codificar.
        :return: Tupla (train_df_transformado, test_df_transformado).
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Copias y validación básica de columnas
        # -------------------------------------------------------------------------------------------------
        train_df: pd.DataFrame = train_df.copy()
        test_df: pd.DataFrame = test_df.copy()
        present_cols = [c for c in cols_to_encode if c in train_df.columns and c in test_df.columns]
        if not present_cols:
            return train_df, test_df

        # -------------------------------------------------------------------------------------------------
        # -- 2: Aseguramos tipo category en las columnas a codificar
        # -------------------------------------------------------------------------------------------------
        for col in present_cols:
            train_df[col] = train_df[col].astype("category")
            test_df[col] = test_df[col].astype("category")

        # -------------------------------------------------------------------------------------------------
        # -- 3: Fit_transform en train y transform en test con LeaveOneOutEncoder
        # -------------------------------------------------------------------------------------------------
        encoder: LeaveOneOutEncoder = LeaveOneOutEncoder()
        train_encoded = encoder.fit_transform(train_df[present_cols], target)
        test_encoded = encoder.transform(test_df[present_cols])

        # -------------------------------------------------------------------------------------------------
        # -- 4: Reconstrucción de dataframes
        # -------------------------------------------------------------------------------------------------
        train_df = pd.concat([train_df.drop(present_cols, axis=1), train_encoded], axis=1)
        test_df = pd.concat([test_df.drop(present_cols, axis=1), test_encoded], axis=1)

        return train_df, test_df

    @staticmethod
    def sum_encoder(train_df: pd.DataFrame, test_df: pd.DataFrame, cols_to_encode: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Codifica variables categóricas utilizando SumEncoder.
        Genera representaciones continuas basadas en suma/agregación de dummy-encodings.

        ➤ Usar cuando: buscas una representación numérica simple y rápida,
          útil para modelos lineales o cuando la granularidad no es crítica.

        :param train_df: DataFrame de entrenamiento (sin columna objetivo).
        :param test_df: DataFrame de prueba (sin columna objetivo).
        :param cols_to_encode: Lista de columnas a codificar.
        :return: Tupla (train_df_transformado, test_df_transformado).
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Copias y validación básica de columnas
        # -------------------------------------------------------------------------------------------------
        train_df: pd.DataFrame = train_df.copy()
        test_df: pd.DataFrame = test_df.copy()
        present_cols = [c for c in cols_to_encode if c in train_df.columns and c in test_df.columns]
        if not present_cols:
            return train_df, test_df

        # -------------------------------------------------------------------------------------------------
        # -- 2: Aplicación de SumEncoder sobre el DataFrame completo (según implementación esperada)
        # -------------------------------------------------------------------------------------------------
        encoder: SumEncoder = SumEncoder(cols=present_cols)
        train_encoded = encoder.fit_transform(train_df)
        test_encoded = encoder.transform(test_df)

        # -------------------------------------------------------------------------------------------------
        # -- 3: Retorno (se asume que encoder devuelve DataFrames completos)
        # -------------------------------------------------------------------------------------------------
        return train_encoded, test_encoded

    @staticmethod
    def target_encoder(train_df: pd.DataFrame, test_df: pd.DataFrame, target: pd.Series, cols_to_encode: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Codifica variables categóricas utilizando TargetEncoder (media del target por categoría).

        ➤ Usar cuando: necesitas un encoding simple y rápido; ten en cuenta riesgo de overfitting.

        :param train_df: DataFrame de entrenamiento (sin columna objetivo).
        :param test_df: DataFrame de prueba (sin columna objetivo).
        :param target: Serie objetivo de entrenamiento.
        :param cols_to_encode: Lista de columnas a codificar.
        :return: Tupla (train_df_transformado, test_df_transformado).
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Copias y validación básica de columnas
        # -------------------------------------------------------------------------------------------------
        train_df: pd.DataFrame = train_df.copy()
        test_df: pd.DataFrame = test_df.copy()
        present_cols = [c for c in cols_to_encode if c in train_df.columns and c in test_df.columns]
        if not present_cols:
            return train_df, test_df

        # -------------------------------------------------------------------------------------------------
        # -- 2: Aseguramos tipo category en las columnas a codificar
        # -------------------------------------------------------------------------------------------------
        for col in present_cols:
            train_df[col] = train_df[col].astype("category")
            test_df[col] = test_df[col].astype("category")

        # -------------------------------------------------------------------------------------------------
        # -- 3: Fit y transform con TargetEncoder
        # -------------------------------------------------------------------------------------------------
        encoder: TargetEncoder = TargetEncoder()
        encoder.fit(train_df[present_cols], target)

        train_df[present_cols] = encoder.transform(train_df[present_cols])
        test_df[present_cols] = encoder.transform(test_df[present_cols])

        # -------------------------------------------------------------------------------------------------
        # -- 4: Retorno
        # -------------------------------------------------------------------------------------------------
        return train_df, test_df

    # </editor-fold>

    # <editor-fold desc="Scalers    ----------------------------------------------------------------------------------------------------------------------------------------------">

    @staticmethod
    def min_max_scaler(train_df: pd.DataFrame, test_df: pd.DataFrame, cols_to_scale: list[str], excluded_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
        """
        Escala los datos entre 0 y 1 utilizando MinMaxScaler.

        ➤ Usar cuando: quieres normalizar datos a un rango fijo, útil para algoritmos sensibles a magnitudes (p. ej. redes neuronales).

        :param train_df: DataFrame de entrenamiento.
        :param test_df: DataFrame de prueba.
        :param cols_to_scale: Columnas a escalar.
        :param excluded_cols: Columnas a mantener sin escalar.
        :return: Tupla (train_scaled_df, test_scaled_df, scaler).
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Copias y separación de columnas excluidas
        # -------------------------------------------------------------------------------------------------
        train_df: pd.DataFrame = train_df.copy()
        test_df: pd.DataFrame = test_df.copy()
        train_excluded = train_df[excluded_cols]
        test_excluded = test_df[excluded_cols]

        # -------------------------------------------------------------------------------------------------
        # -- 2: Aplicación del MinMaxScaler
        # -------------------------------------------------------------------------------------------------
        scaler: MinMaxScaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df[cols_to_scale])
        test_scaled = scaler.transform(test_df[cols_to_scale])

        train_scaled_df = pd.DataFrame(train_scaled, columns=cols_to_scale, index=train_df.index)
        test_scaled_df = pd.DataFrame(test_scaled, columns=cols_to_scale, index=test_df.index)

        # -------------------------------------------------------------------------------------------------
        # -- 3: Concatenar columnas excluidas
        # -------------------------------------------------------------------------------------------------
        train_scaled_df = pd.concat([train_scaled_df, train_excluded], axis=1)
        test_scaled_df = pd.concat([test_scaled_df, test_excluded], axis=1)

        return train_scaled_df, test_scaled_df, scaler

    @staticmethod
    def max_abs_scaler(train_df: pd.DataFrame, test_df: pd.DataFrame, cols_to_scale: list[str], excluded_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, MaxAbsScaler]:
        """
        Escala los datos de forma que el valor absoluto máximo de cada columna sea 1.

        ➤ Usar cuando: los datos contienen valores positivos y negativos y quieres preservar la dispersión relativa.

        :param train_df: DataFrame de entrenamiento.
        :param test_df: DataFrame de prueba.
        :param cols_to_scale: Columnas a escalar.
        :param excluded_cols: Columnas a mantener sin escalar.
        :return: Tupla (train_scaled_df, test_scaled_df, scaler).
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Copias y separación de columnas excluidas
        # -------------------------------------------------------------------------------------------------
        train_df: pd.DataFrame = train_df.copy()
        test_df: pd.DataFrame = test_df.copy()
        train_excluded = train_df[excluded_cols]
        test_excluded = test_df[excluded_cols]

        # -------------------------------------------------------------------------------------------------
        # -- 2: Aplicación del MaxAbsScaler
        # -------------------------------------------------------------------------------------------------
        scaler: MaxAbsScaler = MaxAbsScaler()
        train_scaled = scaler.fit_transform(train_df[cols_to_scale])
        test_scaled = scaler.transform(test_df[cols_to_scale])

        train_scaled_df = pd.DataFrame(train_scaled, columns=cols_to_scale, index=train_df.index)
        test_scaled_df = pd.DataFrame(test_scaled, columns=cols_to_scale, index=test_df.index)

        # -------------------------------------------------------------------------------------------------
        # -- 3: Concatenar columnas excluidas
        # -------------------------------------------------------------------------------------------------
        train_scaled_df = pd.concat([train_scaled_df, train_excluded], axis=1)
        test_scaled_df = pd.concat([test_scaled_df, test_excluded], axis=1)

        return train_scaled_df, test_scaled_df, scaler

    @staticmethod
    def robust_scaler(train_df: pd.DataFrame, test_df: pd.DataFrame, cols_to_scale: list[str], excluded_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
        """
        Escala los datos utilizando estadísticas robustas frente a outliers (mediana y IQR).

        ➤ Usar cuando: los datos contienen outliers que podrían afectar a escaladores clásicos como StandardScaler o MinMaxScaler.

        :param train_df: DataFrame de entrenamiento.
        :param test_df: DataFrame de prueba.
        :param cols_to_scale: Columnas a escalar.
        :param excluded_cols: Columnas a mantener sin escalar.
        :return: Tupla (train_scaled_df, test_scaled_df, scaler).
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Copias y separación de columnas excluidas
        # -------------------------------------------------------------------------------------------------
        train_df: pd.DataFrame = train_df.copy()
        test_df: pd.DataFrame = test_df.copy()
        train_excluded = train_df[excluded_cols]
        test_excluded = test_df[excluded_cols]

        # -------------------------------------------------------------------------------------------------
        # -- 2: Aplicación del RobustScaler
        # -------------------------------------------------------------------------------------------------
        scaler: RobustScaler = RobustScaler()
        train_scaled = scaler.fit_transform(train_df[cols_to_scale])
        test_scaled = scaler.transform(test_df[cols_to_scale])

        train_scaled_df = pd.DataFrame(train_scaled, columns=cols_to_scale, index=train_df.index)
        test_scaled_df = pd.DataFrame(test_scaled, columns=cols_to_scale, index=test_df.index)

        # -------------------------------------------------------------------------------------------------
        # -- 3: Concatenar columnas excluidas
        # -------------------------------------------------------------------------------------------------
        train_scaled_df = pd.concat([train_scaled_df, train_excluded], axis=1)
        test_scaled_df = pd.concat([test_scaled_df, test_excluded], axis=1)

        return train_scaled_df, test_scaled_df, scaler

    @staticmethod
    def standard_scaler(train_df: pd.DataFrame, test_df: pd.DataFrame, cols_to_scale: list[str], excluded_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """
        Escala los datos para que tengan media 0 y desviación estándar 1.

        ➤ Usar cuando: quieres normalizar variables continuas para algoritmos sensibles a la magnitud de los datos (p. ej. regresión, SVM).

        :param train_df: DataFrame de entrenamiento.
        :param test_df: DataFrame de prueba.
        :param cols_to_scale: Columnas a escalar.
        :param excluded_cols: Columnas a mantener sin escalar.
        :return: Tupla (train_scaled_df, test_scaled_df, scaler).
        """

        # -------------------------------------------------------------------------------------------------
        # -- 1: Copias y separación de columnas excluidas
        # -------------------------------------------------------------------------------------------------
        train_df: pd.DataFrame = train_df.copy()
        test_df: pd.DataFrame = test_df.copy()
        train_excluded = train_df[excluded_cols]
        test_excluded = test_df[excluded_cols]

        # -------------------------------------------------------------------------------------------------
        # -- 2: Aplicación del StandardScaler
        # -------------------------------------------------------------------------------------------------
        scaler: StandardScaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df[cols_to_scale])
        test_scaled = scaler.transform(test_df[cols_to_scale])

        train_scaled_df = pd.DataFrame(train_scaled, columns=cols_to_scale, index=train_df.index)
        test_scaled_df = pd.DataFrame(test_scaled, columns=cols_to_scale, index=test_df.index)

        # -------------------------------------------------------------------------------------------------
        # -- 3: Concatenar columnas excluidas
        # -------------------------------------------------------------------------------------------------
        train_scaled_df = pd.concat([train_scaled_df, train_excluded], axis=1)
        test_scaled_df = pd.concat([test_scaled_df, test_excluded], axis=1)

        return train_scaled_df, test_scaled_df, scaler

    # </editor-fold>
