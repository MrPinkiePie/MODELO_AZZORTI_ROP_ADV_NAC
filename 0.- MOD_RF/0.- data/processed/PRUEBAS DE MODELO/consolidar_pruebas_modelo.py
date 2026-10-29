#%%
"""
Concatena todos los archivos 'df_*' (csv o parquet) de esta misma carpeta
(una corrida por archivo: referencia de 0.- MOD_RF y talla de 0.- MOD_TLL)
en un solo parquet para cargar al Power BI (DSB) y comparar todas las corridas.

Los entrenar.py exportan su df_final tal cual; AQUI (no en los entrenar.py) se
agregan las columnas del formato estandar que espera el DSB -- el mismo
TST_REF_PREV de los scripts de tallas ECU/BOL -- sin borrar las originales:
COMPANIA, LINEA, MODELO, VERSION, CAMP_ORIGEN/DESTINO, CODI_VENT, DESCRIPCION,
Q_50, Q_REAL, Q_RG3, Q_MKT, Q_FALT, TOT_PED_*, SPLIT, ORDEN_SPLIT, TIPO, etc.

Agrega ademas:
- ARCHIVO_ORIGEN: de que archivo (corrida) viene cada fila.
- NIVEL: 'REFERENCIA' o 'TALLA'. Los entrenar.py ya la escriben; si un archivo
  viejo no la trae, se deduce del nombre (los de talla terminan en 'TALLAS').
- ES_COMUN: True si la referencia (campaña + NOMBRE_SAP) esta en TODOS los
  archivos. Filtrar ES_COMUN=True compara corridas sobre el mismo universo.
"""
import sys
from pathlib import Path  # manipular rutas de forma OOP

import numpy as np
import pandas as pd

ruta_pruebas = Path(__file__).resolve().parent
archivo_salida = ruta_pruebas / 'consolidado_pruebas_modelo.parquet'

# config_proyecto.py de la raiz del proyecto (parents: PRUEBAS, processed, 0.- data, 0.- MOD_x, PROYECTO)
sys.path.append(str(ruta_pruebas.parents[3]))
import config_proyecto as cfg

MODELO = 'ADV_NAC' if hasattr(cfg, 'PAIS_ORIGEN') else 'NAC_NAC'  # solo ADV_NAC define pais origen

# columnas que pide el DSB (Table.TransformColumnTypes falla si falta alguna)
COLUMNAS_DSB = ['COMPANIA', 'PAIS', 'LINEA', 'MODELO', 'VERSION', 'TIPO_ZONA_DESTINO', 'CAMP_ORIGEN',
                'CAMP_DESTINO', 'CODI_VENT', 'DESCRIPCION', 'LIM_INF', 'Q_10', 'Q_20', 'Q_30', 'Q_40',
                'Q_50', 'Q_60', 'Q_70', 'Q_80', 'Q_90', 'LIM_SUP', 'Q_RG3', 'TOT_PED_EST', 'Q_MKT',
                'TOT_PED_MKT', 'Q_REAL', 'Q_FALT', 'TOT_PED_REAL', 'FALT_ANUN', 'PERCENTIL_ELEGIDO',
                'COMENTARIO', 'SUBGRUPO', 'GRUPO', 'NOMBRE_SAP', 'PRECIO_N', 'ORDEN_SPLIT', 'SPLIT',
                'PRECIO_PROMEDIO_COMPETIDORES_DESTINO', 'NUM_PRODUCTOS_POR_CATEGORIAS_ORIGEN',
                'QUINTIL_PREC_ADV', 'TIPO']


def primera_disponible(df, candidatas, por_defecto=np.nan):
    """Devuelve la primera columna de candidatas que exista en df (cada modelo usa nombres distintos)."""
    for col in candidatas:
        if col in df.columns:
            return df[col]
    return por_defecto


def a_formato_dsb(df):
    """
    Agrega (o completa) las columnas estandar del DSB a partir de las que exporta
    cada entrenar.py. Referencia y talla, ADV_NAC y NAC_NAC traen nombres distintos
    para lo mismo: aqui se unifican. No borra ninguna columna original.
    """
    df = df.copy()
    es_talla = df['NIVEL'].iloc[0] == 'TALLA'

    df['COMPANIA'] = 'AZZ'
    df['LINEA'] = cfg.LINEA
    df['MODELO'] = MODELO
    df['VERSION'] = df['COMENTARIO']
    df['TIPO_ZONA_DESTINO'] = 'N'

    # en ADV_NAC la campaña destino es CAMP y la origen (advance) CAMP_ADV
    df['CAMP_DESTINO'] = primera_disponible(df, ['CAMP_DESTINO', 'CAMP'])
    df['CAMP_ORIGEN'] = primera_disponible(df, ['CAMP_ORIGEN', 'CAMP_ADV'])

    # la referencia agrupa varias tallas: su "codigo" es el NOMBRE_SAP
    df['CODI_VENT'] = df['CODI_VENT'] if es_talla else df['NOMBRE_SAP']
    df['DESCRIPCION'] = primera_disponible(df, ['NOMBRE_SID']) if es_talla else df['NOMBRE_SAP']

    # consolidar_resultados (evaluacion.py) renombra Q_50 -> y_pred y el real -> y_real
    df['Q_50'] = df['y_pred']
    df['Q_REAL'] = df['y_real']

    df['Q_RG3'] = primera_disponible(df, ['FACT_ADV_RGN', 'FACTOR_RCP_ORIGEN'])  # base del pronostico
    df['Q_MKT'] = primera_disponible(df, ['FAC_MKT', 'FACTOR_MKT'])
    df['Q_FALT'] = primera_disponible(df, ['Q_FALT', 'Q_FALT_N'])
    df['TOT_PED_REAL'] = primera_disponible(df, ['TOT_N'], por_defecto=1)
    df['PRECIO_N'] = primera_disponible(df, ['PRECIO_N', 'PRECIO_N_DESTINO'])

    # constantes del formato (mismos valores que TST_REF_PREV de los scripts ECU/BOL)
    df['TOT_PED_EST'] = 1
    df['TOT_PED_MKT'] = 1
    df['FALT_ANUN'] = 'NO'
    df['PERCENTIL_ELEGIDO'] = 'Q_50'

    # SPLIT estandar TRAIN/VALIDACION en ambos proyectos: CONJUNTO ya viene asi
    # de consolidar_resultados (ADV usa TRAIN/TEST por dentro, NAC_NAC TRAIN/VALIDACION)
    df['SPLIT'] = df['CONJUNTO']
    df['ORDEN_SPLIT'] = np.where(df['SPLIT'] == 'TRAIN', 1, 2)

    df['TIPO'] = primera_disponible(df, ['TIPO', 'TIPO_REF'])

    # variables propias de un solo modelo: se crean vacias si este no las trae
    for col in COLUMNAS_DSB:
        if col not in df.columns:
            df[col] = np.nan
    return df


archivos = sorted(f for f in ruta_pruebas.glob('df_*') if f.suffix in ('.csv', '.parquet'))
print(f'====== {MODELO} | ARCHIVOS ENCONTRADOS: {len(archivos)} ======')

dfs = []
for archivo in archivos:
    if archivo.suffix == '.csv':
        df = pd.read_csv(archivo, encoding='cp1252', low_memory=False)
    else:
        df = pd.read_parquet(archivo)
    df['ARCHIVO_ORIGEN'] = archivo.stem
    if 'NIVEL' not in df.columns:
        df['NIVEL'] = 'TALLA' if archivo.stem.upper().endswith('TALLAS') else 'REFERENCIA'
    df = a_formato_dsb(df)
    print(f'{archivo.name}: {len(df):,} filas | NIVEL={df["NIVEL"].iloc[0]}')
    dfs.append(df)

df_consolidado = pd.concat(dfs, ignore_index=True).copy()  # copy desfragmenta (evita PerformanceWarning)

# llave de referencia (ya unificada en CAMP_DESTINO + NOMBRE_SAP)
df_consolidado['REF_KEY'] = df_consolidado['CAMP_DESTINO'].astype(str) + '_' + df_consolidado['NOMBRE_SAP'].astype(str)

# TIPO solo viene en las filas de referencia (ADV_NAC): las tallas heredan el de su referencia
tipo_por_ref = (df_consolidado[df_consolidado['NIVEL'] == 'REFERENCIA']
                .dropna(subset=['TIPO']).drop_duplicates('REF_KEY').set_index('REF_KEY')['TIPO'])
df_consolidado['TIPO'] = df_consolidado['TIPO'].fillna(df_consolidado['REF_KEY'].map(tipo_por_ref))

# tallas nuevas (NAC_NAC: la talla no existia en la campaña origen de su referencia):
# heredan CAMP_ORIGEN de su referencia y su Q_RG3 (venta en origen) es 0
es_talla = df_consolidado['NIVEL'] == 'TALLA'
camp_origen_por_ref = (df_consolidado[~es_talla].dropna(subset=['CAMP_ORIGEN'])
                       .drop_duplicates('REF_KEY').set_index('REF_KEY')['CAMP_ORIGEN'])
df_consolidado.loc[es_talla, 'CAMP_ORIGEN'] = (df_consolidado.loc[es_talla, 'CAMP_ORIGEN']
                                               .fillna(df_consolidado.loc[es_talla, 'REF_KEY'].map(camp_origen_por_ref)))
df_consolidado.loc[es_talla, 'Q_RG3'] = df_consolidado.loc[es_talla, 'Q_RG3'].fillna(0)

# ES_COMUN: referencias presentes en todos los archivos (interseccion)
n_archivos = df_consolidado['ARCHIVO_ORIGEN'].nunique()
archivos_por_ref = df_consolidado.groupby('REF_KEY')['ARCHIVO_ORIGEN'].transform('nunique')
df_consolidado['ES_COMUN'] = archivos_por_ref == n_archivos
df_consolidado = df_consolidado.drop(columns='REF_KEY')

# columnas del DSB primero, el resto (variables de cada modelo) despues
otras = [c for c in df_consolidado.columns if c not in COLUMNAS_DSB]
df_consolidado = df_consolidado[COLUMNAS_DSB + otras]

print(f'====== CONSOLIDADO: {len(df_consolidado):,} FILAS | '
      f'{df_consolidado["ES_COMUN"].mean():.1%} EN REFERENCIAS COMUNES A LOS {n_archivos} ARCHIVOS ======')
print(df_consolidado.groupby(['NIVEL', 'SPLIT', 'ARCHIVO_ORIGEN']).size().to_string())
vacias = [c for c in COLUMNAS_DSB if df_consolidado[c].isna().all()]
print(f'Columnas del DSB sin datos en este modelo (quedan vacias): {vacias}')

# columnas de texto con tipos mezclados entre archivos (ej. CAMP int en uno,
# str en otro) rompen PyArrow al guardar: se dejan como texto
for col in df_consolidado.select_dtypes(include='object').columns:
    df_consolidado[col] = df_consolidado[col].astype('string')

df_consolidado.to_parquet(archivo_salida, index=False)
print(f'====== GUARDADO EN: {archivo_salida} ======')
