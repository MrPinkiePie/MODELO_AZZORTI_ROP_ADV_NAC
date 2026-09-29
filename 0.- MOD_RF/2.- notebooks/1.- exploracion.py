"""
EDA (analisis exploratorio) de las tablas de destino (NAC) y origen (ADV) de
ROP_ADV_NAC, antes del cruce por campana desfasada. El flujo productivo ya
migrado vive en 3.- src/models/entrenar.py -- este notebook es para inspeccion
rapida de shapes/columnas/valores unicos al explorar una tabla nueva.
"""
#%% Librerias
import sys
from pathlib import Path  # manipular rutas de forma OOP

sys.path.append(str(Path(__file__).resolve().parents[1] / "3.- src"))
from config import PAIS_DESTINO, PAIS_ORIGEN, LINEA
from data.cargar_datos import (load_1_1, load_vehiculo_estrategia, mapear_categorias_por_subgrupo,
                                leer_fct_adv_gp1, calcular_factores_adv_gp1, revisar_calidad_fct_adv_gp1)
from data.eda import resumen_nulos, resumen_cardinalidad, graficar_boxplot_factor_rcp
import matplotlib.pyplot as plt

camp_inicio, camp_fin = 201801, 202613  # mismo rango que 3.- src/models/entrenar.py -- bump camp_fin cuando haya campanas nuevas

#%% Destino (NAC)
df_destino = load_1_1(pais=PAIS_DESTINO, camp_inicio=camp_inicio, camp_end=camp_fin)
print(df_destino.shape)
df_destino.head()

#%%
df_destino['LINEA_FIN'].value_counts()

#%% EDA rapido de df_destino: nulos y cardinalidad, previo al cruce por campana desfasada
resumen_nulos(df_destino)
resumen_cardinalidad(df_destino)

#%% Origen (ADV)
df_origen_tzo = load_1_1(pais=PAIS_ORIGEN, camp_inicio=camp_inicio, camp_end=camp_fin, linea=LINEA)
print(df_origen_tzo.shape)
df_origen_tzo.head()

#%% EDA rapido de df_origen_tzo: nulos y cardinalidad
resumen_nulos(df_origen_tzo)
resumen_cardinalidad(df_origen_tzo)

#%%
df_raw_fct_adv, archivos_fct_adv = leer_fct_adv_gp1(pais=PAIS_ORIGEN)
revisar_calidad_fct_adv_gp1(df_raw_fct_adv, pais=PAIS_ORIGEN)  # solo informativo, no filtra ni detiene nada
df_fct_adv = calcular_factores_adv_gp1(df_raw_fct_adv, pais=PAIS_ORIGEN)
df_fct_adv.describe()

#%% EDA rapido de df_fct_adv: nulos, cardinalidad y boxplot de FACT_ADV_RGN (posibles atipicos)
resumen_nulos(df_fct_adv)
resumen_cardinalidad(df_fct_adv)
graficar_boxplot_factor_rcp(df_fct_adv, columna_factor='FACT_ADV_RGN')
plt.show()

#%% Vehiculo/estrategia
df_veh_est_destino = load_vehiculo_estrategia(pais=PAIS_DESTINO, camp_inicio=camp_inicio, camp_end=camp_fin)
df_veh_est_destino['VEHICULO'].value_counts()

#%% EDA rapido de df_veh_est_destino: nulos y cardinalidad
resumen_nulos(df_veh_est_destino)
resumen_cardinalidad(df_veh_est_destino)

#%%
df_veh_est_origen = load_vehiculo_estrategia(pais=PAIS_ORIGEN, camp_inicio=camp_inicio, camp_end=camp_fin)
df_veh_est_origen['VEHICULO'].value_counts()

#%% EDA rapido de df_veh_est_origen: nulos y cardinalidad
resumen_nulos(df_veh_est_origen)
resumen_cardinalidad(df_veh_est_origen)

#%% CATEGORIAS por SUBGRUPO (antes se leia de un excel, ver mapear_categorias_por_subgrupo)
df_destino = mapear_categorias_por_subgrupo(df_destino, linea=LINEA)
df_destino[['SUBGRUPO', 'CATEGORIAS']].drop_duplicates().head()
