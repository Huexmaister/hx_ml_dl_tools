from hx_hyperparameters_builder.abstract_hyperparameter_builder import AbstractHyperparameterBuilder
from typing import Dict, Union

# <editor-fold desc="MACHINE LEARNING --------------------------------------------------------------------------------------------------------------------------------------------">

class HxLightgbmHyperparameterBuilder(AbstractHyperparameterBuilder):
    def __init__(self, custom_hyperparameters: Dict[str, Union[int, float]] | None = None):

        # -------------------------------------------------------------------------------
        # -- 1: Llamo a la clase padre con la configuracion concreta del modelo
        # -------------------------------------------------------------------------------

        # ---- 1.1: Hago el call y asigno los valores por defecto a las propiedades recogidas en la clase padre
        super().__init__(
            {'max_depth': (1, 4), 'learning_rate': (0.005, 0.2), 'n_estimators': (100, 3000), 'reg_alpha': (0, 3), 'reg_lambda': (0, 3)},
            ["learning_rate", "min_split_gain"],
            {'max_depth': (1, 15), 'learning_rate': (0.001, 0.5), 'n_estimators': (100, 8000), 'reg_alpha': (0, 9), 'reg_lambda': (0, 9)}
        )

        # -------------------------------------------------------------------------------
        # -- 2: Obtengo los custom_hyperparameters
        # -------------------------------------------------------------------------------

        # ---- 2.1: Almaceno en una propiedad los hiperparametros que ha definido el usuario
        self.custom_hyperparameters: Dict[str, Union[int, float]] | None = custom_hyperparameters

    def validate_malformation_ranges(self):
        pass

    def validate_and_get_hyperparams(self) -> Dict[str, Union[int, float]]:
        """
        Metodo que va a devolver los custom_hyperparameters si no son None y va a lanzar una excepcion en caso contrario
        :return:
        """

        # -- Planteo la excepción
        if (self.custom_hyperparameters is None or
                not isinstance(self.custom_hyperparameters, dict) or
                not all(isinstance(k, str) for k in self.custom_hyperparameters.keys()) or
                not all(isinstance(v, (int, float)) for v in self.custom_hyperparameters.values())):
            raise ValueError("custom_hyperparameters must be Dict[str, Union[int, float]]. Example: {'max_depth': 3, 'learning_rate': 0.01}")

        # -- Los retorno si están bien
        return self.custom_hyperparameters

class HxXgboostHyperparameterBuilder(AbstractHyperparameterBuilder):
    def __init__(self, custom_hyperparameters: Dict[str, Union[int, float]] | None = None):

        # -------------------------------------------------------------------------------
        # -- 1: Llamo a la clase padre con la configuracion concreta del modelo
        # -------------------------------------------------------------------------------

        # ---- 1.1: Hago el call y asigno los valores por defecto a las propiedades recogidas en la clase padre
        super().__init__(
            {'max_depth': (1, 10), 'learning_rate': (0.005, 0.2), 'n_estimators': (100, 3000), 'reg_alpha': (0, 3), 'reg_lambda': (0, 3)},
            ["learning_rate", "min_split_gain"],
            {'max_depth': (1, 15), 'learning_rate': (0.001, 0.5), 'n_estimators': (100, 8000), 'reg_alpha': (0, 9), 'reg_lambda': (0, 9)}
        )

        # -------------------------------------------------------------------------------
        # -- 2: Obtengo los custom_hyperparameters
        # -------------------------------------------------------------------------------

        # ---- 2.1: Almaceno en una propiedad los hiperparametros que ha definido el usuario
        self.custom_hyperparameters: Dict[str, Union[int, float]] | None = custom_hyperparameters

    def validate_malformation_ranges(self):
        pass

    def validate_and_get_hyperparams(self) -> Dict[str, Union[int, float]]:
        """
        Metodo que va a devolver los custom_hyperparameters si no son None y va a lanzar una excepcion en caso contrario
        :return:
        """

        # -- Planteo la excepción
        if (self.custom_hyperparameters is None or
                not isinstance(self.custom_hyperparameters, dict) or
                not all(isinstance(k, str) for k in self.custom_hyperparameters.keys()) or
                not all(isinstance(v, (int, float)) for v in self.custom_hyperparameters.values())):
            raise ValueError("custom_hyperparameters must be Dict[str, Union[int, float]]. Example: {'max_depth': 3, 'learning_rate': 0.01}")

        # -- Los retorno si están bien
        return self.custom_hyperparameters

class HxCatBoostHyperparameterBuilder(AbstractHyperparameterBuilder):
    def __init__(self, custom_hyperparameters: Dict[str, Union[int, float]] | None = None):

        # -------------------------------------------------------------------------------
        # -- 1: Llamo a la clase padre con la configuracion concreta del modelo
        # -------------------------------------------------------------------------------

        # ---- 1.1: Hago el call y asigno los valores por defecto a las propiedades recogidas en la clase padre
        super().__init__(
            {'iterations': (100, 900), 'max_depth': (1, 10), 'learning_rate': (0.005, 0.2), 'l2_leaf_reg': (0, 3), 'penalties_coefficient': (0, 3)},
            ["learning_rate"],
            {'iterations': (100, 5000), 'max_depth': (1, 15), 'learning_rate': (0.001, 0.5), 'l2_leaf_reg': (0, 9), 'penalties_coefficient': (0, 9)}
        )

        # -------------------------------------------------------------------------------
        # -- 2: Obtengo los custom_hyperparameters
        # -------------------------------------------------------------------------------

        # ---- 2.1: Almaceno en una propiedad los hiperparametros que ha definido el usuario
        self.custom_hyperparameters: Dict[str, Union[int, float]] | None = custom_hyperparameters

    def validate_malformation_ranges(self):
        pass

    def validate_and_get_hyperparams(self) -> Dict[str, Union[int, float]]:
        """
        Metodo que va a devolver los custom_hyperparameters si no son None y va a lanzar una excepcion en caso contrario
        :return:
        """

        # -- Planteo la excepción
        if (self.custom_hyperparameters is None or
                not isinstance(self.custom_hyperparameters, dict) or
                not all(isinstance(k, str) for k in self.custom_hyperparameters.keys()) or
                not all(isinstance(v, (int, float)) for v in self.custom_hyperparameters.values())):
            raise ValueError("custom_hyperparameters must be Dict[str, Union[int, float]]. Example: {'max_depth': 3, 'learning_rate': 0.01}")

        # -- Los retorno si están bien
        return self.custom_hyperparameters

class HxSvmHyperparameterBuilder(AbstractHyperparameterBuilder):
    def __init__(self, custom_hyperparameters: Dict[str, Union[int, float]] | None = None):

        # -------------------------------------------------------------------------------
        # -- 1: Llamo a la clase padre con la configuracion concreta del modelo
        # -------------------------------------------------------------------------------

        # ---- 1.1: Hago el call y asigno los valores por defecto a las propiedades recogidas en la clase padre
        super().__init__(
            {'C': (0.01, 10), 'gamma': (0.1, 0.8), 'degree': (2, 4)},
            ["C", "gamma"],
            {'C': (0.001, 10), 'gamma': (0.0, 1), 'degree': (1, 10)}
        )

        # -------------------------------------------------------------------------------
        # -- 2: Obtengo los custom_hyperparameters
        # -------------------------------------------------------------------------------

        # ---- 2.1: Almaceno en una propiedad los hiperparametros que ha definido el usuario
        self.custom_hyperparameters: Dict[str, Union[int, float]] | None = custom_hyperparameters

    def validate_malformation_ranges(self):
        pass

    def validate_and_get_hyperparams(self) -> Dict[str, Union[int, float]]:
        """
        Metodo que va a devolver los custom_hyperparameters si no son None y va a lanzar una excepcion en caso contrario
        :return:
        """

        # -- Planteo la excepción
        if (self.custom_hyperparameters is None or
                not isinstance(self.custom_hyperparameters, dict) or
                not all(isinstance(k, str) for k in self.custom_hyperparameters.keys()) or
                not all(isinstance(v, (int, float)) for v in self.custom_hyperparameters.values())):
            raise ValueError("custom_hyperparameters must be Dict[str, Union[int, float]]. Example: {'max_depth': 3, 'learning_rate': 0.01}")

        # -- Los retorno si están bien
        return self.custom_hyperparameters

# </editor-fold>

# <editor-fold desc="DEEP LEARNING -----------------------------------------------------------------------------------------------------------------------------------------------">

class HxDnnHyperparameterBuilder(AbstractHyperparameterBuilder):
    def __init__(self, custom_hyperparameters: Dict[str, Union[int, float]] | None = None):

        # -------------------------------------------------------------------------------
        # -- 1: Llamo a la clase padre con la configuracion concreta del modelo
        # -------------------------------------------------------------------------------

        # ---- 1.1: Hago el call y asigno los valores por defecto a las propiedades recogidas en la clase padre
        super().__init__(
            {'units': (248, 1024), 'dropout': (0, 0.3), 'epochs': (20, 250), 'batch_size': (16, 1024), 'learning_rate': (0.00001, 0.001),
             'regularization': (0, 0.001), 'layers_design': (0, 2.2), 'layers_number': (1, 8)},
            ["learning_rate", "dropout", "regularization"],
            {'units': (4, 4096), 'dropout': (0, 0.4), 'epochs': (10, 400), 'batch_size': (8, 2048), 'learning_rate': (0.000005, 0.5),
             'regularization': (0, 0.005), 'layers_design': (0, 4), 'layers_number': (1, 9)}
        )

        # -------------------------------------------------------------------------------
        # -- 2: Obtengo los custom_hyperparameters
        # -------------------------------------------------------------------------------

        # ---- 2.1: Almaceno en una propiedad los hiperparametros que ha definido el usuario
        self.custom_hyperparameters: Dict[str, Union[int, float]] | None = custom_hyperparameters

    def validate_malformation_ranges(self):
        pass

    def validate_and_get_hyperparams(self) -> Dict[str, Union[int, float]]:
        """
        Metodo que va a devolver los custom_hyperparameters si no son None y va a lanzar una excepcion en caso contrario
        :return:
        """

        # -- Planteo la excepción
        if (self.custom_hyperparameters is None or
                not isinstance(self.custom_hyperparameters, dict) or
                not all(isinstance(k, str) for k in self.custom_hyperparameters.keys()) or
                not all(isinstance(v, (int, float)) for v in self.custom_hyperparameters.values())):
            raise ValueError("custom_hyperparameters must be Dict[str, Union[int, float]]. Example: {'max_depth': 3, 'learning_rate': 0.01}")

        # -- Los retorno si están bien
        return self.custom_hyperparameters

class HxCnn1dHyperparameterBuilder(AbstractHyperparameterBuilder):
    def __init__(self, custom_hyperparameters: Dict[str, Union[int, float]] | None = None):

        # -------------------------------------------------------------------------------
        # -- 1: Llamo a la clase padre con la configuracion concreta del modelo
        # -------------------------------------------------------------------------------

        # ---- 1.1: Hago el call y asigno los valores por defecto a las propiedades recogidas en la clase padre
        super().__init__(
            {'filters': (8, 128), 'kernel_size': (3, 5), 'pool_size': (0, 4), 'dropout': (0, 0.2), 'epochs': (20, 50), 'batch_size': (24, 128),
             'learning_rate': (0.00001, 0.001), 'regularization': (0, 0.001), 'layers_design': (0, 2.2), 'layers_number': (0, 2), 'dense_aditional_layers': (0, 2),
             'dense_aditional_units': (32, 128)},
            ["learning_rate", "dropout", "regularization"],
            {'filters': (4, 1024), 'kernel_size': (2, 9), 'pool_size': (0, 4), 'dropout': (0, 0.3), 'epochs': (10, 120), 'batch_size': (16, 1024),
             'learning_rate': (0.000005, 0.5), 'regularization': (0, 0.005), 'layers_design': (0, 4), 'layers_number': (0, 3), 'dense_aditional_layers': (0, 3),
             'dense_aditional_units': (32, 128)}
        )

        # -------------------------------------------------------------------------------
        # -- 2: Obtengo los custom_hyperparameters
        # -------------------------------------------------------------------------------

        # ---- 2.1: Almaceno en una propiedad los hiperparametros que ha definido el usuario
        self.custom_hyperparameters: Dict[str, Union[int, float]] | None = custom_hyperparameters

    def validate_malformation_ranges(self):
        pass

    def validate_and_get_hyperparams(self) -> Dict[str, Union[int, float]]:
        """
        Metodo que va a devolver los custom_hyperparameters si no son None y va a lanzar una excepcion en caso contrario
        :return:
        """

        # -- Planteo la excepción
        if (self.custom_hyperparameters is None or
                not isinstance(self.custom_hyperparameters, dict) or
                not all(isinstance(k, str) for k in self.custom_hyperparameters.keys()) or
                not all(isinstance(v, (int, float)) for v in self.custom_hyperparameters.values())):
            raise ValueError("custom_hyperparameters must be Dict[str, Union[int, float]]. Example: {'max_depth': 3, 'learning_rate': 0.01}")

        # -- Los retorno si están bien
        return self.custom_hyperparameters

class HxLstmHyperparameterBuilder(AbstractHyperparameterBuilder):
    def __init__(self, custom_hyperparameters: Dict[str, Union[int, float]] | None = None):

        # -------------------------------------------------------------------------------
        # -- 1: Llamo a la clase padre con la configuracion concreta del modelo
        # -------------------------------------------------------------------------------

        # ---- 1.1: Hago el call y asigno los valores por defecto a las propiedades recogidas en la clase padre
        super().__init__(
            {'timesteps': (3, 7), 'filters': (8, 256), 'kernel_size': (3, 5), 'dropout': (0, 0.2), 'epochs': (20, 120), 'batch_size': (24, 512),
             'learning_rate': (0.00001, 0.001), 'regularization': (0, 0.001), 'layers_design': (0, 4), 'layers_number': (0, 3), 'dense_aditional_layers': (0, 3),
             'dense_aditional_units': (32, 512), 'lstm_aditional_layers': (0, 3), 'lstm_aditional_units': (32, 128)},
            ["learning_rate", "dropout", "regularization"],
            {'timesteps': (3, 7), 'filters': (6, 512), 'kernel_size': (2, 6), 'dropout': (0, 0.3), 'epochs': (15, 150), 'batch_size': (16, 1024),
             'learning_rate': (0.000005, 0.5), 'regularization': (0, 0.005), 'layers_design': (0, 4), 'layers_number': (0, 4), 'dense_aditional_layers': (0, 3),
             'dense_aditional_units': (24, 648), 'lstm_aditional_layers': (0, 3), 'lstm_aditional_units': (24, 164)}
        )

        # -------------------------------------------------------------------------------
        # -- 2: Obtengo los custom_hyperparameters
        # -------------------------------------------------------------------------------

        # ---- 2.1: Almaceno en una propiedad los hiperparametros que ha definido el usuario
        self.custom_hyperparameters: Dict[str, Union[int, float]] | None = custom_hyperparameters

    def validate_malformation_ranges(self):
        pass

    def validate_and_get_hyperparams(self) -> Dict[str, Union[int, float]]:
        """
        Metodo que va a devolver los custom_hyperparameters si no son None y va a lanzar una excepcion en caso contrario
        :return:
        """

        # -- Planteo la excepción
        if (self.custom_hyperparameters is None or
                not isinstance(self.custom_hyperparameters, dict) or
                not all(isinstance(k, str) for k in self.custom_hyperparameters.keys()) or
                not all(isinstance(v, (int, float)) for v in self.custom_hyperparameters.values())):
            raise ValueError("custom_hyperparameters must be Dict[str, Union[int, float]]. Example: {'max_depth': 3, 'learning_rate': 0.01}")

        # -- Los retorno si están bien
        return self.custom_hyperparameters

class HxHybridHyperparameterBuilder(AbstractHyperparameterBuilder):
    def __init__(self, custom_hyperparameters: Dict[str, Union[int, float]] | None = None):

        # -------------------------------------------------------------------------------
        # -- 1: Llamo a la clase padre con la configuracion concreta del modelo
        # -------------------------------------------------------------------------------

        # ---- 1.1: Hago el call y asigno los valores por defecto a las propiedades recogidas en la clase padre
        super().__init__(
            {'filters': (8, 256), 'kernel_size': (3, 12), 'pool_size': (0, 1), 'dropout': (0, 0.2), 'epochs': (20, 70), 'batch_size': (128, 512),
             'learning_rate': (0.00001, 0.001), 'regularization': (0, 0.001), 'layers_design': (0, 4), 'layers_number': (0, 2), 'dense_aditional_layers': (0, 2),
             'dense_aditional_units': (32, 128), 'lstm_aditional_layers': (0, 2), 'lstm_aditional_units': (32, 128)},
            ["learning_rate", "dropout", "regularization"],
            {'filters': (4, 1024), 'kernel_size': (2, 15), 'pool_size': (0, 4), 'dropout': (0, 0.3), 'epochs': (10, 120), 'batch_size': (24, 1024),
             'learning_rate': (0.000005, 0.5), 'regularization': (0, 0.005), 'layers_design': (0, 4), 'layers_number': (0, 4), 'dense_aditional_layers': (0, 3),
             'dense_aditional_units': (32, 128), 'lstm_aditional_layers': (0, 3), 'lstm_aditional_units': (24, 164)}
        )

        # -------------------------------------------------------------------------------
        # -- 2: Obtengo los custom_hyperparameters
        # -------------------------------------------------------------------------------

        # ---- 2.1: Almaceno en una propiedad los hiperparametros que ha definido el usuario
        self.custom_hyperparameters: Dict[str, Union[int, float]] | None = custom_hyperparameters

    def validate_malformation_ranges(self):
        pass

    def validate_and_get_hyperparams(self) -> Dict[str, Union[int, float]]:
        """
        Metodo que va a devolver los custom_hyperparameters si no son None y va a lanzar una excepcion en caso contrario
        :return:
        """

        # -- Planteo la excepción
        if (self.custom_hyperparameters is None or
                not isinstance(self.custom_hyperparameters, dict) or
                not all(isinstance(k, str) for k in self.custom_hyperparameters.keys()) or
                not all(isinstance(v, (int, float)) for v in self.custom_hyperparameters.values())):
            raise ValueError("custom_hyperparameters must be Dict[str, Union[int, float]]. Example: {'max_depth': 3, 'learning_rate': 0.01}")

        # -- Los retorno si están bien
        return self.custom_hyperparameters

class HxTransformerHyperparameterBuilder(AbstractHyperparameterBuilder):
    def __init__(self, custom_hyperparameters: Dict[str, Union[int, float]] | None = None):

        # -------------------------------------------------------------------------------
        # -- 1: Llamo a la clase padre con la configuracion concreta del modelo
        # -------------------------------------------------------------------------------

        # ---- 1.1: Hago el call y asigno los valores por defecto a las propiedades recogidas en la clase padre
        super().__init__(
            {'num_blocks': (1, 3), 'sequence_length': (1, 4), 'buffer_size': (100, 1500), 'd_model': (32, 256), 'num_heads': (1, 8), 'ff_dim': (1, 4),
             'embedding_size': (32, 128), 'dropout_rate': (0.1, 0.4), 'batch_size': (8, 32), 'epochs': (1, 100), 'learning_rate': (0.00001, 0.0001), 'units': (1, 50),
             'feature_dim': (1, 1), 'regularization': (0.0, 0.2), 'layers_design': (0, 4), 'layers_number': (1, 3), 'dense_aditional_layers': (0, 2),
             'dense_aditional_units': (24, 128)},
            ["learning_rate", "dropout_rate", "regularization"],
            {'num_blocks': (1, 6), 'sequence_length': (1, 5), 'buffer_size': (100, 1500), 'd_model': (32, 256), 'num_heads': (1, 8), 'ff_dim': (1, 4),
             'embedding_size': (32, 128), 'dropout_rate': (0.1, 0.4), 'batch_size': (8, 32), 'epochs': (1, 100), 'learning_rate': (0.000001, 0.001), 'units': (1, 50),
             'feature_dim': (1, 1), 'regularization': (0.0, 0.2), 'layers_design': (0, 4.25), 'layers_number': (1, 3), 'dense_aditional_layers': (0, 2),
             'dense_aditional_units': (24, 128)}
        )

        # -------------------------------------------------------------------------------
        # -- 2: Obtengo los custom_hyperparameters
        # -------------------------------------------------------------------------------

        # ---- 2.1: Almaceno en una propiedad los hiperparametros que ha definido el usuario
        self.custom_hyperparameters: Dict[str, Union[int, float]] | None = custom_hyperparameters

    def validate_malformation_ranges(self):
        pass

    def validate_and_get_hyperparams(self) -> Dict[str, Union[int, float]]:
        """
        Metodo que va a devolver los custom_hyperparameters si no son None y va a lanzar una excepcion en caso contrario
        :return:
        """

        # -- Planteo la excepción
        if (self.custom_hyperparameters is None or
                not isinstance(self.custom_hyperparameters, dict) or
                not all(isinstance(k, str) for k in self.custom_hyperparameters.keys()) or
                not all(isinstance(v, (int, float)) for v in self.custom_hyperparameters.values())):
            raise ValueError("custom_hyperparameters must be Dict[str, Union[int, float]]. Example: {'max_depth': 3, 'learning_rate': 0.01}")

        # -- Los retorno si están bien
        return self.custom_hyperparameters

# </editor-fold>