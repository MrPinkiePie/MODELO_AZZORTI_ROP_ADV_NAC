"""
Inspecciona el resultado de 3.- src/models/entrenar.py a nivel referencia
(dat_ref, exportado como 0.- data/processed/datos_estimar_5.csv en el PASO 6
de entrenar.py): pivot tables, distribucion de CAMBIO_PAG_LADO, etc.

Requiere haber corrido entrenar.py al menos una vez antes (este notebook NO
vuelve a correr la carga/limpieza/features, solo lee el CSV ya exportado).
Lo que se valide aca ya esta reflejado en 3.- src/features/construir_features.py.
"""
#%% Librerias
import sys
from pathlib import Path  # manipular rutas de forma OOP

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "3.- src"))
from config import DATA_PROCESSED, DATA_INTERIM

dat_ref = pd.read_csv(DATA_PROCESSED / 'datos_estimar_5.csv', encoding='cp1252')

#%%
print(dat_ref.shape)
dat_ref[['CAMP_NOMBRE_SAP', 'RATIO_FACTOR', 'CAMBIO_PAG_LADO', 'TIPO_REF']].head()

#%% Distribucion de RATIO_FACTOR por CAMBIO_PAG_LADO
dat_ref.pivot_table(index='CAMBIO_PAG_LADO', values='RATIO_FACTOR', aggfunc='mean').reset_index()

#%%
dat_ref.pivot_table(index='CAMBIO_PAG_LADO', values='RATIO_FACTOR', aggfunc='median').reset_index()

#%% Cuantas referencias son NUEVA vs CON_HISTORIA
dat_ref['TIPO_REF'].value_counts(normalize=True)

#%%
#===============================================
# EDA DE RATIO_FACTOR (equivalente a FACTOR_RCP en MODELO_AZZORTI_ROP_NAC_NAC,
# ver src/data/eda.py de ese proyecto) -- usa dat_ref_5.parquet, exportado al
# FINAL DEL PASO 5 de entrenar.py (dat_ref recien con RATIO_FACTOR calculado,
# TODAVIA con los posibles atipicos que el PASO 6 filtra despues), para poder
# ver esos atipicos aca en vez de perderlos antes de llegar a este notebook.
#===============================================
from data.eda import (resumen_nulos, resumen_cardinalidad, crear_anio_camp,
                       graficar_histograma_factor_rcp, graficar_boxplot_factor_rcp,
                       graficar_factor_rcp_por_campana, graficar_factor_rcp_por_anio,
                       graficar_tendencia_factor_rcp, graficar_ranking_factor_por_estrategia,
                       resumen_factor_por_nombre_sap)

dat_ref_eda = pd.read_parquet(DATA_INTERIM / 'dat_ref_5.parquet')
print(dat_ref_eda.shape)
dat_ref_eda.head()

#%% Nulos por columna
resumen_nulos(dat_ref_eda)

#%% Varianza / valores unicos
resumen_cardinalidad(dat_ref_eda)

#%% separa CAMP (formato AAAACC) en anio y num_campana
dat_ref_eda = crear_anio_camp(dat_ref_eda)

#%% Histograma de RATIO_FACTOR (la Y), con posibles outliers
graficar_histograma_factor_rcp(dat_ref_eda, columna_factor='RATIO_FACTOR',
                                columna_camp='CAMP', columna_descripcion='NOMBRE_SAP')
plt.show()

#%% Boxplot de RATIO_FACTOR en agregado
graficar_boxplot_factor_rcp(dat_ref_eda, columna_factor='RATIO_FACTOR')
plt.show()

#%% RATIO_FACTOR por campaña (comparando entre años)
graficar_factor_rcp_por_campana(dat_ref_eda, columna_factor='RATIO_FACTOR')
plt.show()

graficar_factor_rcp_por_campana(dat_ref_eda, columna_factor='RATIO_FACTOR', agregacion='sum')
plt.show()

#%% RATIO_FACTOR por año
graficar_factor_rcp_por_anio(dat_ref_eda, columna_factor='RATIO_FACTOR')
plt.show()

#%% Tendencia en agregado (todas las campañas/años juntos)
graficar_tendencia_factor_rcp(dat_ref_eda, columna_factor='RATIO_FACTOR', agregacion='sum')
plt.show()

graficar_tendencia_factor_rcp(dat_ref_eda, columna_factor='RATIO_FACTOR', agregacion='mean')
plt.show()

#%% Tendencia por GRUPO
graficar_tendencia_factor_rcp(dat_ref_eda, columna_factor='RATIO_FACTOR', por='GRUPO')
plt.show()

#%% Boxplot de RATIO_FACTOR por GRUPO
graficar_boxplot_factor_rcp(dat_ref_eda, columna_factor='RATIO_FACTOR', por='GRUPO')
plt.show()

#%% Ranking de RATIO_FACTOR por ESTR_MIX (mezcla de estrategia origen/destino
# -- equivalente a ESTRATEGIA en NAC_NAC, que dat_ref no trae a nivel referencia)
graficar_ranking_factor_por_estrategia(dat_ref_eda, columna_factor='RATIO_FACTOR', columna_grupo='ESTR_MIX')
plt.show()

#%% Resumen de RATIO_FACTOR por NOMBRE_SAP (promedio por anio, promedio y
# desviacion general, cantidad de ESTR_MIX distintos en los que salio)
tabla_nombre_sap = resumen_factor_por_nombre_sap(dat_ref_eda, columna_factor='RATIO_FACTOR',
                                                  columna_estrategia='ESTR_MIX')

#%%
#===============================================
# RELEVANCIA DE VARIABLES: informacion mutua (MI) de las variables YA usadas
# en features['num']/['cat'] de entrenar.py PASO 9, contra LOG_RATIO_FACTOR.
# Usa df_limpio.parquet (export final de entrenar.py, PASO 9): ya trae
# LOG_RATIO_FACTOR, CAMP_SEN/COS y RATIO_PAG_NAC_ADV calculados y ya sin las
# infladas de origen (PASO 7), igual que MODELO_AZZORTI_ROP_NAC_NAC.
#===============================================
import numpy as np

from features.construir_features import obtener_features_candidatas
from config import PAIS_ORIGEN, FACTORES_REGIONALES_POR_PAIS
from data.eda import ranking_informacion_mutua

df_limpio = pd.read_parquet(DATA_PROCESSED / 'df_limpio.parquet')

# solo TRAIN: elegir variables mirando test seria leakage
df_mi = df_limpio[df_limpio['SPLIT'] == 'TRAIN'].copy()

# set completo de candidatas, el mismo que usa entrenar.py con usar_boruta=True
# (features/construir_features.py). Los factores regionales salen de config.py
# segun PAIS_ORIGEN (COL o PER), asi no revienta al cambiar de pais
candidatas = obtener_features_candidatas(FACTORES_REGIONALES_POR_PAIS[PAIS_ORIGEN])
features_actuales_num = candidatas['num']
features_actuales_cat = candidatas['cat']

# red de seguridad: avisa y descarta las que no esten en el parquet (ej.
# generado con otro PAIS_ORIGEN) en vez de reventar con KeyError
faltantes = [c for c in features_actuales_num + features_actuales_cat if c not in df_mi.columns]
if faltantes:
    print(f'OJO, no estan en df_limpio.parquet (se omiten): {faltantes}')
features_actuales_num = [c for c in features_actuales_num if c in df_mi.columns]
features_actuales_cat = [c for c in features_actuales_cat if c in df_mi.columns]

# mutual_info_regression no acepta NaN/inf: se descartan esas filas
columnas_mi = features_actuales_num + features_actuales_cat + ['LOG_RATIO_FACTOR']
n_antes = len(df_mi)
df_mi = df_mi[columnas_mi].replace([np.inf, -np.inf], np.nan).dropna()
print(f'====== FILAS PARA MI: {len(df_mi):,} de {n_antes:,} (se quitan NaN/inf) ======')

#%% Informacion mutua vs LOG_RATIO_FACTOR (mayor valor = mas relevante)
tabla_mi, fig, ax = ranking_informacion_mutua(
    df_mi, columna_target='LOG_RATIO_FACTOR',
    candidatas_num=features_actuales_num, candidatas_cat=features_actuales_cat)
plt.show()
