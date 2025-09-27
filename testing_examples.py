# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ---------------------------------------            REGRESSION            ---------------------------------------
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

from hx_ml_dl_tools_examples import DiabetesExample, CaliforniaHousingExample

# ----------------------------------------------------------------
# -- 1: Ejemplo de regresion con Diabetes
# ----------------------------------------------------------------

# DiabetesExample().execute_example()

# ----------------------------------------------------------------
# -- 2: Ejemplo de regresion con CaliforniaHousing
# ----------------------------------------------------------------

# CaliforniaHousingExample().execute_example()

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ----------------------------------            BINARY CLASSIFICATION            ---------------------------------
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

from hx_ml_dl_tools_examples import BreastCancerExample, CreditFraudExample

# ----------------------------------------------------------------
# -- 3: Ejemplo de clasificacion binaria con BreastCancer
# ----------------------------------------------------------------

# BreastCancerExample().execute_example()

# ----------------------------------------------------------------
# -- 4: Ejemplo de clasificacion binaria con CreditFraud
# ----------------------------------------------------------------

# CreditFraudExample(50000).execute_example()


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# --------------------------------            MULTICLASS CLASSIFICATION            -------------------------------
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

from hx_ml_dl_tools_examples import IrisMulticlassExample, DigitsMulticlassExample, WineQualityMulticlassExample

# ----------------------------------------------------------------
# -- 5: Ejemplo de clasificacion multiclase con IrisMulticlassExample
# ----------------------------------------------------------------

# IrisMulticlassExample().execute_example()

# ----------------------------------------------------------------
# -- 6: Ejemplo de clasificacion multiclase con WineQualityMulticlassExample
# ----------------------------------------------------------------

# WineQualityMulticlassExample().execute_example()

# ----------------------------------------------------------------
# -- 7: Ejemplo de clasificacion multiclase con DigitsMulticlassExample
# ----------------------------------------------------------------

DigitsMulticlassExample().execute_example()
