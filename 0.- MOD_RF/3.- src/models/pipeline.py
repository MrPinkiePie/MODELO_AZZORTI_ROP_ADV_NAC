"""
Define la estructura del sklearn.pipeline.Pipeline: preprocesamiento
(escalado de numericas + CatBoostEncoder de categoricas) encadenado con el
RandomForestRegressor. Mismo patron que
MODELO_AZZORTI_ROP_NAC_NAC/.../models/pipeline.py -- no lee archivos ni hace
fit aqui, eso vive en entrenar.py.
"""
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def construir_pipeline(features, params, seed=42):
    """
    Arma el Pipeline de sklearn (ColumnTransformer con escalado de numericas
    + CatBoostEncoder de categoricas, seguido del RandomForestRegressor) sin
    hacer fit. params son los hiperparametros del RandomForest (ej. los que
    devuelve entrenamiento_utils.ejecutar_estudio_optuna).
    """
    preprocesador = ColumnTransformer([
        ('num', StandardScaler(), features['num']),
        ('cat', ce.CatBoostEncoder(cols=features['cat'], random_state=seed), features['cat'])
    ])

    pipe = Pipeline(steps=[
        ('prep', preprocesador),
        ('modelo', RandomForestRegressor(**params, bootstrap=True, random_state=seed))
    ])

    return pipe
