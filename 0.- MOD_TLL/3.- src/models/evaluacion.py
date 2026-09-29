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


# ===========================================================================
# MODELO POR TALLAS (propio de 0.- MOD_TLL) -- mismas funciones que
# MODELO_AZZORTI_ROP_NAC_NAC/0.- MOD_TLL, con la llave de este proyecto
# ===========================================================================

# percentiles que entrega 0.- MOD_RF (ver calcular_percentiles); en su
# df_final el Q_50 viene renombrado como y_pred (ver consolidar_resultados)
COLUMNAS_PERCENTILES = ['LIM_INF', 'Q_10', 'Q_20', 'Q_30', 'Q_40', 'Q_50',
                        'Q_60', 'Q_70', 'Q_80', 'Q_90', 'LIM_SUP']


def rebalancear_participacion(df, col_pred='PRED_PART', nueva_col='PART_P_50'):
    """
    Normaliza la participacion predicha para que sume 1 dentro de cada
    referencia (CAMP_NOMBRE_SAP): el RF predice cada talla por separado y
    la suma no tiene por que dar 1. Si la referencia suma 0, reparte en
    partes iguales.
    """
    df = df.copy()
    por_referencia = df.groupby('CAMP_NOMBRE_SAP')[col_pred]
    suma = por_referencia.transform('sum')
    n_tallas = por_referencia.transform('size')
    df[nueva_col] = np.where(suma > 0, df[col_pred] / suma, 1 / n_tallas)
    return df


def repartir_percentiles_referencia(df, df_ref, col_part='PART_P_50', claves=('CAMP', 'NOMBRE_SAP')):
    """
    Cruza cada talla con el pronostico de su referencia (df_final de
    0.- MOD_RF, ya con y_pred renombrado a Q_50) por claves (inner) y
    reparte cada percentil segun col_part. En ADV_NAC, CAMP ya es la campaña
    destino (NAC), la misma que usa MOD_RF.

    Devuelve (df_cruzado, percentiles): percentiles es un dict con la misma
    forma que calcular_percentiles ({'LIM_INF': array, 'Q_10': array, ...}),
    para entrar directo a agregar_percentiles_y_error. df_cruzado trae las
    columnas de la referencia con prefijo REF_ (REF_Q_50, REF_y_real, ...).
    """
    claves = list(claves)
    df_ref = df_ref[claves + COLUMNAS_PERCENTILES + ['y_real']].copy()
    df_ref = df_ref.rename(columns={c: f'REF_{c}' for c in COLUMNAS_PERCENTILES + ['y_real']})

    # llaves con el mismo tipo en ambos lados (el csv de MOD_RF puede leerlas distinto)
    df = df.copy()
    for d in (df, df_ref):
        d[claves[0]] = d[claves[0]].astype(int)
        d[claves[1]] = d[claves[1]].astype(str)

    df_cruzado = pd.merge(df, df_ref, on=claves, how='inner')
    print(f'====== CRUCE CON PRONOSTICO DE REFERENCIA (inner): {len(df)} -> {len(df_cruzado)} FILAS ======')

    percentiles = {c: (df_cruzado[col_part] * df_cruzado[f'REF_{c}']).to_numpy()
                   for c in COLUMNAS_PERCENTILES}
    return df_cruzado, percentiles
