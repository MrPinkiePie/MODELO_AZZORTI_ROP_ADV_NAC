"""
Funciones para evaluar el pipeline ya entrenado (ver pipeline.py/entrenar.py):
predicciones por arbol y sus percentiles, importancia de variables, y
consolidacion de resultados train/test. Mismo patron que
MODELO_AZZORTI_ROP_NAC_NAC/.../models/evaluacion.py, adaptado a la escala de
ROP_ADV_NAC (LOG_RATIO_FACTOR, sin el log-shift +0.0001*1000 de NAC_NAC --
ver entrenamiento_utils.invert_y).
"""
import sys
from pathlib import Path  # manipular rutas de forma OOP

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.entrenamiento_utils import invert_y
from data.mis_funciones import svs_sbr_por_grupo


def predicciones_por_arbol(pipe, X, base_df, ratio_col='FACTOR_RCP_ORIGEN', y_mode='log_ratio'):
    """
    Predice con cada arbol del RandomForest del pipeline sobre X (en escala
    LOG_RATIO_FACTOR) y revierte la transformacion a escala real con
    invert_y, multiplicando por base_df[ratio_col]. base_df debe estar
    alineado por posicion con X (mismas filas, mismo orden -- ej. el propio
    train/valid del que salio X via armar_x_y).
    Devuelve un DataFrame de forma (n_filas, n_arboles), con el indice de X.
    """
    rf = pipe.named_steps['modelo']
    X_enc = pipe.named_steps['prep'].transform(X)
    predicciones_log = np.array([estimador.predict(X_enc) for estimador in rf.estimators_]).T

    base_df = base_df.reset_index(drop=True)
    predicciones = np.column_stack([
        invert_y(predicciones_log[:, i], base_df, ratio_col=ratio_col, mode=y_mode)
        for i in range(predicciones_log.shape[1])
    ])
    return pd.DataFrame(predicciones, index=X.index)


def calcular_percentiles(matriz_predicciones, percentiles=(5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95)):
    """
    Calcula percentiles por fila sobre una matriz de predicciones (una
    columna por arbol). Devuelve un dict {'Q_30': array, 'Q_40': array, ...};
    los percentiles 5 y 95 se devuelven como 'LIM_INF'/'LIM_SUP' (limites de
    un intervalo que excluye el 10% mas extremo de los arboles).
    """
    etiquetas = {5: 'LIM_INF', 95: 'LIM_SUP'}
    return {
        etiquetas.get(p, f'Q_{p}'): np.percentile(matriz_predicciones, p, axis=1)
        for p in percentiles
    }


def obtener_importancia_variables(pipe, features):
    """
    Arma un DataFrame con la importancia de cada variable segun el
    RandomForest del pipeline, ordenado de mayor a menor.
    """
    importancia = pd.DataFrame({
        'variable': features['all'],
        'importancia': pipe.named_steps['modelo'].feature_importances_
    }).sort_values('importancia', ascending=False).reset_index(drop=True)
    return importancia


def agregar_percentiles_y_error(df, percentiles, col_real='FACTOR_RCP_DESTINO'):
    """
    Agrega al DataFrame las columnas de percentiles (dict de calcular_percentiles)
    y el error de la mediana (Q_50) contra col_real: MAE_fila y MAPE_fila.
    """
    df = df.copy()
    for nombre, valores in percentiles.items():
        df[nombre] = valores
    df['MAE_fila'] = abs(df['Q_50'] - df[col_real])
    df['MAPE_fila'] = abs(df['Q_50'] - df[col_real]) / df[col_real]
    return df


def consolidar_resultados(df_train_pred, df_test_pred, comentario, col_real='FACTOR_RCP_DESTINO'):
    """
    Une train y test en un solo DataFrame, marca el conjunto (TRAIN/VALIDACION),
    renombra columnas a y_real/y_pred, calcula Q70_Q30 y agrega el comentario
    de la corrida.
    """
    df_train_pred = df_train_pred.copy()
    df_test_pred = df_test_pred.copy()
    df_train_pred['CONJUNTO'] = 'TRAIN'
    df_test_pred['CONJUNTO'] = 'VALIDACION'

    df_final = pd.concat([df_train_pred, df_test_pred]).reset_index(drop=True)
    df_final = df_final.rename(columns={col_real: 'y_real', 'Q_50': 'y_pred'})
    df_final['Q70_Q30'] = (df_final['Q_70'] / df_final['Q_30']).round(2)
    df_final['COMENTARIO'] = comentario

    return df_final


def analizar_svs_sbr_por_bin_ratio(df, col_ratio='Q70_Q30', corte_fijos=False):
    """
    Agrupa el DataFrame en bins de col_ratio (cortes fijos de negocio o
    cuantiles automaticos con qcut) y calcula SVS/SBR (sobre y_pred/y_real,
    ver consolidar_resultados) por bin. Devuelve el resultado en formato
    largo (RATIO, METRICA, VALOR), listo para graficar con
    visualization.graficos.graficar_svs_sbr_por_bin. Copiada de
    MODELO_AZZORTI_ROP_NAC_NAC.
    """
    df = df.copy()

    if corte_fijos:
        condicion = [
            df[col_ratio].between(0, 1.5), df[col_ratio].between(1.5, 2),
            df[col_ratio].between(2, 2.5), df[col_ratio].between(2.5, 3),
            df[col_ratio].between(3, 4), df[col_ratio] > 4,
        ]
        etiquetas = ['0-1.5', '1.5-2', '2-2.5', '2.5-3', '3-4', '>4']
        df['RATIO'] = np.select(condicion, etiquetas)
    else:
        df['RATIO'] = pd.qcut(df[col_ratio], q=6, duplicates='drop', precision=2)
        df['RATIO'] = df['RATIO'].astype(str)

    print(df.groupby('RATIO').size())

    resultado = svs_sbr_por_grupo(df, 'RATIO', col_pred='y_pred', col_real='y_real')

    resultado_largo = resultado.melt(
        id_vars='RATIO',
        value_vars=['SVS', 'SBR'],
        var_name='METRICA',
        value_name='VALOR',
    )
    return resultado_largo
