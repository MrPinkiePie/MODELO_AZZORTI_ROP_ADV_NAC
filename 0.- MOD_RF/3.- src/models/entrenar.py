"""
Orquesta el entrenamiento completo de ROP_ADV_NAC: carga destino (NAC) y
origen (ADV), limpia cada lado, cruza por campana desfasada
(DESFASE_CAMPANAS), construye features, agrupa a nivel referencia
(CAMP_NOMBRE_SAP), busca hiperparametros con Optuna, entrena el pipeline
final y lo guarda en 4.- models/. Migrado de "0.- MOD_0.py" (PASO 1-10) y de
"SBR_SVR_RF.py" (PASO 11 en adelante: fit final, percentiles, metricas y
export -- mismo enfoque que entrenar.py de MODELO_AZZORTI_ROP_NAC_NAC).

El pronostico resultante queda a nivel de referencia (CAMP_NOMBRE_SAP, el
codigo SAP del producto), agregando todas las tallas (CODI_VENT) del producto
en una sola fila (ver PASO 5). El reparto por talla vive en 0.- MOD_TLL, que
carga dos salidas de este script: interim/dat.parquet (PASO 4, nivel talla)
y el csv de PRUEBAS DE MODELO (PASO 15). TODO: todavia no existe
predecir.py (aplicar el .pkl guardado sobre campanas nuevas).
"""
#%%
import sys
from pathlib import Path  # manipular rutas de forma OOP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (PAIS_DESTINO, PAIS_ORIGEN, FACTORES_REGIONALES_POR_PAIS, LINEA, DESFASE_CAMPANAS, NOMBRE_PKL,
                    MODELS_DIR, PRUEBAS_MODELO_DIR, DATA_INTERIM, DATA_PROCESSED,
                    CAMP_INI, CAMP_FIN, CORTE_TRAIN_TEST, COMENTARIO_REG,
                    TLL_ROOT, PRUEBAS_MODELO_TLL_DIR)
from data.cargar_datos import (load_1_1, load_vehiculo_estrategia, load_inflados,
                                leer_fct_adv_gp1, calcular_factores_adv_gp1, revisar_calidad_fct_adv_gp1,
                                mapear_categorias_por_subgrupo, load_cons_prod)
from data.mis_funciones import descomponer_camp, metricas, svs_sbr, svs_sbr_por_grupo, log, resumen_filas, titulo_paso
from data.limpiar_datos import (cruzar_vehiculo_estrategia, marcar_atipicos, marcar_faltantes,
                                 filtrar_linea, filtrar_pagina_valida, filtrar_positivos,
                                 filtrar_surtido_sorpresa, marcar_posibles_atipicos_ratio)
from features.construir_features import (construir_features_competencia, calcular_camp_mas_desfase,
                                          reconstruir_clave_desfasada,
                                          cruzar_origen_destino, calcular_factor_nacional, pag_der,
                                          calcular_cambio_pagina_lado, calcular_ubicacion_pagina,
                                          calcular_ratio_precio,
                                          agrupar_por_referencia, unir_extras_por_referencia,
                                          calcular_ratio_factor, calcular_resumen_factores_regionales,
                                          marcar_tipo_referencia, calcular_exposicion,
                                          calcular_variables_ciclicas_campana,calcular_momentum_subgrupo,
                                          obtener_features_candidatas)
from models.entrenamiento_utils import (dividir_train_test_por_corte, armar_x_y,
                                         seleccionar_features_boruta, construir_objetivo_optuna,
                                         ejecutar_estudio_optuna)
from models.pipeline import construir_pipeline
from models.evaluacion import (predicciones_por_arbol, calcular_percentiles, obtener_importancia_variables,
                                agregar_percentiles_y_error, consolidar_resultados,
                                analizar_svs_sbr_por_bin_ratio)
from visualization.graficos import graficar_svs_sbr_por_bin

pais_destino = PAIS_DESTINO   # demanda nacional a pronosticar
pais_origen = PAIS_ORIGEN     # demanda advance que anticipa al NAC
linea = LINEA
desfase_campanas = DESFASE_CAMPANAS

# paises, campañas, corte y comentario se editan en config_proyecto.py (raiz
# del proyecto): los comparte con 0.- MOD_TLL, que reparte este pronostico
camp_inicio = CAMP_INI
camp_fin = CAMP_FIN

corte_train_test = CORTE_TRAIN_TEST
seed = 42


usar_boruta = False  # True para reducir variables con Boruta antes de Optuna 
max_iter=50

usar_optuna = True  # False usa los hiperparametros fijos de abajo (mas rapido); True corre Optuna 
n_trials = 50       # bajarlo reduce el tiempo de corrida, a costa de una busqueda menos fina
usar_ponderacion_temporal = True # True para que Optuna busque un decay_rate y aplique ponderacion exponencial por antiguedad



comentario = COMENTARIO_REG
nombre_modelo = NOMBRE_PKL
guardar_pkl = True   # True reemplaza el .pkl anterior en 4.- models/

# Set FIJO elegido a mano (se usa si usar_boruta=False).
FEATURES_FIJAS = {
    'target': 'LOG_RATIO_FACTOR',
    'num': ['RATIO_PRECIO', 'FACT_ADV_RGN', 'FACTOR_MEDIANA', 'FACTOR_MAX', 'FACTOR_MIN',
            'PRECIO_PROMEDIO_COMPETIDORES_ORIGEN', 'NUM_PRODUCTOS_POR_CATEGORIAS_ORIGEN',
            'RATIO_PROMEDIO_COMPETIDORES_PRECIO_ORIGEN',
            'PRECIO_PROMEDIO_COMPETIDORES_DESTINO', 'NUM_PRODUCTOS_POR_CATEGORIAS_DESTINO',
            'PAG_NAC_NEW', 'PAG_ADV_IN',
            'CAMP_SEN', 'CAMP_COS','MOMENTUM_SUBGRUPO_LAG_3'],
    'cat': ['NUM_CAMP_MIX', 'SUBGRUPO', 'CATEGORIAS'],
}



#%%
#===============================================
# PASO 1: DESTINO (NAC) -- cargar, features de competencia, limpiar
#===============================================
titulo_paso('PASO 1: DESTINO (NAC) -- cargar, features de competencia, limpiar')

bitacora_destino = []
bitacora_origen = []

def contar_referencias(df, es_origen=False):
    if 'CAMP_NOMBRE_SAP' in df.columns:
        return df['CAMP_NOMBRE_SAP'].nunique()
    
    camp_col = 'CAMP_ADV' if (es_origen and 'CAMP_ADV' in df.columns) else 'CAMP'
    
    if camp_col in df.columns and 'NOMBRE_SAP' in df.columns:
        return len(df[[camp_col, 'NOMBRE_SAP']].drop_duplicates())
        
    return 'No aplica'

def add_track_dest(df, comentario):
    diff = float('nan') if not bitacora_destino else len(df) - bitacora_destino[-1]['FILAS']
    refs = contar_referencias(df, es_origen=False)
    bitacora_destino.append({
        'COMENTARIO': comentario, 
        'FILAS': len(df), 
        'DIFERENCIA': diff, 
        'REFERENCIAS_UNICAS': refs,
        'NOTAS': ''
    })

def add_track_orig(df, comentario):
    diff = float('nan') if not bitacora_origen else len(df) - bitacora_origen[-1]['FILAS']
    refs = contar_referencias(df, es_origen=True)
    nota = 'Aún no se ha cruzado con el catálogo de productos (CON_PRD) para saber el NOMBRE_SAP' if refs == 'No aplica' else ''
    bitacora_origen.append({
        'COMENTARIO': comentario, 
        'FILAS': len(df), 
        'DIFERENCIA': diff, 
        'REFERENCIAS_UNICAS': refs,
        'NOTAS': nota
    })

df_destino = load_1_1(pais=pais_destino, camp_inicio=camp_inicio, camp_end=camp_fin)  # sin filtro de linea todavia (ver nota en load_1_1)
add_track_dest(df_destino, 'Cargue de 1.1 (Destino)')

df_destino = mapear_categorias_por_subgrupo(df_destino, linea=linea)

# Features de competencia calculadas ANTES de filtrar linea (igual que el
# original): NUM_PRODUCTOS_POR_CATEGORIAS_PAGINA_DESTINO cuenta competidores
# de TODAS las lineas en esa pagina, no solo RIN.
df_destino = construir_features_competencia(
    df_destino, col_camp='CAMP', col_pagina='PAG_NAC_NEW', col_precio='PRECIO_N',
    col_linea='LINEA_FIN', sufijo='DESTINO', cols_extra_pagina=['PAIS'])
resumen_filas(df_destino, 'features de competencia creadas')

# filtrar_linea ya imprime su propio '🧹 FILTRO ...: N FILAS' -- no se
# duplica con resumen_filas (misma info, mas simple de leer)
df_destino = filtrar_linea(df_destino, linea)
add_track_dest(df_destino, f'filtrando solo {linea}')

# load_vehiculo_estrategia ya imprime '✅ ... | N filas' por dentro
df_veh_est_destino = load_vehiculo_estrategia(pais=pais_destino, camp_inicio=camp_inicio, camp_end=camp_fin)
# cruzar_vehiculo_estrategia ya imprime su propio '🧹 FILTRO VEHICULO [...]: N FILAS'
df_destino = cruzar_vehiculo_estrategia(df_destino, df_veh_est_destino,
                                         vehiculos_validos=['NAL', 'NALADV', 'NAL/NALADV'])
add_track_dest(df_destino, 'filtrando solo NAL,NALADV,NAL/NALADV')

# marcar_atipicos/marcar_faltantes ya imprimen cada una su '🧹 FILTRO VOLAR ...: N FILAS'
df_inflados_destino = load_inflados(pais=pais_destino, tipo_zona='N')
df_destino = marcar_atipicos(df_destino, df_inflados_destino, volar_atipicos=True)
add_track_dest(df_destino, f'filtrando INFLADOS en destino {pais_destino}')

df_destino = marcar_faltantes(df_destino, volar_faltantes=True)
add_track_dest(df_destino, 'filtrando solo FALTANTES')

#%%
#===============================================
# PASO 2: ORIGEN (ADV) -- factor regional (1.2) + precio/pagina (1.1) + vehiculo
#===============================================
titulo_paso('PASO 2: ORIGEN (ADV) -- factor regional (1.2) + precio/pagina (1.1) + vehiculo')
# 2.1: factores regionales de advance (1.2). Se calcula ya aca CAMP_MAS_3 (la
# campana de destino equivalente), pero CAMP_CODI_VENT se deja SIN desfasar
# todavia -- el join de la seccion 2.4 (con precio/pagina/vehiculo) todavia es
# entre dos tablas de origen, ambas sin desfasar.
df_raw_fct_adv, archivos_fct_adv = leer_fct_adv_gp1(pais=pais_origen)
revisar_calidad_fct_adv_gp1(df_raw_fct_adv, pais=pais_origen)  # solo informativo, no filtra ni detiene nada
df_fct_adv = calcular_factores_adv_gp1(df_raw_fct_adv, pais=pais_origen)
print(f'✅ {pais_origen} 1.2 FCT_ADV_GP1: {len(archivos_fct_adv)} archivos | {len(df_fct_adv)} filas')
df_fct_adv = calcular_camp_mas_desfase(df_fct_adv, col_camp='CAMP', desfase_campanas=desfase_campanas)

# 2.2: precio/pagina/cantidad advance, de la misma tabla 1.1 que carga
# destino pero del pais origen -- filtra linea ANTES de construir features
# (a diferencia de destino), igual que el original.
df_origen_tzo = load_1_1(pais=pais_origen, camp_inicio=camp_inicio, camp_end=camp_fin, linea=linea)
add_track_orig(df_origen_tzo, 'Cargue de 1.1 (Origen)')
n_antes = len(df_origen_tzo)
df_origen_tzo = df_origen_tzo.drop_duplicates(['PAIS', 'CAMP', 'CODI_VENT'], keep='first')
add_track_orig(df_origen_tzo, 'Sin duplicados PAIS+CAMP+CODI_VENT')
resumen_filas(df_origen_tzo, 'sin duplicados PAIS+CAMP+CODI_VENT', antes=n_antes)
# filtrar_pagina_valida ya imprime su propio '🧹 FILTRO ...: N FILAS'
df_origen_tzo = filtrar_pagina_valida(df_origen_tzo, col_pagina='PAG_ADV_NEW', max_pag=300)
add_track_orig(df_origen_tzo, 'filtrando pagina valida origen')

df_precio_pagina_origen = pd.DataFrame({
    'CAMP_CODI_VENT': df_origen_tzo['CAMP_CODI_VENT'],
    'Q_PEDI_ADV_IN': df_origen_tzo['Q_PEDI_A'],
    'TOT_PEDI_ADV_IN': df_origen_tzo['TOT_A'],
    'PRECIO_ADV_IN': df_origen_tzo['PRECIO_A'],
    'URL_ADV_IN': df_origen_tzo['URL_ADV'],
    'PAG_ADV_IN': df_origen_tzo['PAG_ADV_NEW'],
    'Q_MKT_ADV_IN': df_origen_tzo['Q_MKT_A'],
})

# 2.3: vehiculo/estrategia del origen, cruzado por la MISMA clave origen
# (CAMP_CODI_VENT sin desfasar todavia -- el desfase ya vive en df_fct_adv).
# load_vehiculo_estrategia ya imprime '✅ ... | N filas' por dentro
df_veh_est_origen = load_vehiculo_estrategia(pais=pais_origen, camp_inicio=camp_inicio, camp_end=camp_fin)
df_precio_pagina_origen = cruzar_vehiculo_estrategia(
    df_precio_pagina_origen, df_veh_est_origen, vehiculos_validos=['ADV', 'NALADV', 'CATALG', 'ADV/NALADV'])
df_precio_pagina_origen = df_precio_pagina_origen.rename(columns={'VEHICULO': 'VEH_ADV', 'ESTRATEGIA': 'ESTRA_ADV'})

# 2.4: unimos el factor regional con precio/pagina/vehiculo -- ambos con
# CAMP_CODI_VENT de origen SIN desfasar todavia (misma campana real de COL).
df_origen = pd.merge(df_fct_adv, df_precio_pagina_origen, how='inner', on='CAMP_CODI_VENT')
add_track_orig(df_origen, 'cruce factor regional (1.2) + precio/pagina (inner)')
resumen_filas(df_origen, 'cruce factor regional (1.2) + precio/pagina/vehiculo (inner)', antes=len(df_fct_adv))
df_origen = df_origen.rename(columns={'CAMP': 'CAMP_ADV'})
# reconstruye CAMP_CODI_VENT usando CAMP_MAS_3 (ya calculado en 2.1): a partir
# de aca la clave queda alineada con la campana de destino, para el cruce
# origen/destino de PASO 3.
df_origen = reconstruir_clave_desfasada(df_origen, col_camp_mas=f'CAMP_MAS_{desfase_campanas}', col_codi_vent='CODI_VENT')

df_origen = filtrar_positivos(df_origen, ['PRECIO_ADV_IN', 'Q_PEDI_ADV_IN', 'Q_MKT_ADV_IN'])
add_track_orig(df_origen, 'filtrando mayores a cero precio cantidad pedida origen')

con_prod = load_cons_prod(columnas=('CODI_VENT', 'NOMBRE_SAP', 'SUBGRUPO', 'LINEA_FIN'))
n_antes = len(df_origen)
df_origen = pd.merge(df_origen, con_prod, how='inner', on='CODI_VENT')
add_track_orig(df_origen, 'Uniendo con cod prod (inner)')
resumen_filas(df_origen, 'cruce con CON_PRD (NOMBRE_SAP/SUBGRUPO, inner)', antes=n_antes)
df_origen = mapear_categorias_por_subgrupo(df_origen, linea=linea)

df_origen = construir_features_competencia(
    df_origen, col_camp='CAMP_ADV', col_pagina='PAG_ADV_IN', col_precio='PRECIO_ADV_IN',
    col_linea='LINEA_FIN', sufijo='ORIGEN')
resumen_filas(df_origen, 'features de competencia creadas')

# columnas que solo servian para construir las features de competencia de
# origen -- destino ya trae sus propios GRUPO/SUBGRUPO/NOMBRE_SAP/LINEA_FIN
df_origen = df_origen.drop(columns=['CATEGORIAS', 'NOMBRE_SAP', 'SUBGRUPO', 'LINEA_FIN'])

#%%
#===============================================
# PASO 3: CRUCE ORIGEN/DESTINO Y LIMPIEZA CONJUNTA
#===============================================
titulo_paso('PASO 3: CRUCE ORIGEN/DESTINO Y LIMPIEZA CONJUNTA')
add_track_orig(df_origen, 'Luego se une con tabla de destino')
dat = cruzar_origen_destino(df_destino, df_origen, col_clave='CAMP_CODI_VENT')
add_track_dest(dat, f'Uniendo con adv {pais_origen} (inner)')

dat = filtrar_surtido_sorpresa(dat)
add_track_dest(dat, 'filtrando solo SURTIDOS y SORPRESA')
dat = calcular_factor_nacional(dat)

dat = filtrar_pagina_valida(dat, col_pagina='PAG_NAC_NEW', max_pag=300)
dat = filtrar_positivos(dat, ['PRECIO_N', 'Q_MKT_N', 'Q_PEDI_N'])
dat = filtrar_positivos(dat, ['FAC_NAC', 'FACT_ADV_RGN'])
add_track_dest(dat, 'filtrando solo FAC NAC Y FAC ADV >0 y precios >0')

dat['N_SKU'] = 1
dat = calcular_ratio_precio(dat, col_destino='PRECIO_N', col_origen='PRECIO_ADV_IN')
dat['CAMP_NOMBRE_SAP'] = dat['CAMP'].astype(str) + '_' + dat['NOMBRE_SAP'].astype(str)
dat['Q_MKT_N'] = dat['Q_MKT_N'] / dat['TOT_PED_MKT_NAC']

#%%
#===============================================
# PASO 4: FEATURES DE PAGINA Y MEZCLA ORIGEN/DESTINO -- a nivel talla, antes de agregar
#===============================================
titulo_paso('PASO 4: FEATURES DE PAGINA Y MEZCLA ORIGEN/DESTINO -- a nivel talla, antes de agregar')
dat = pag_der(dat, 'PAG_ADV_IN')
dat = pag_der(dat, 'PAG_NAC_NEW')
dat = calcular_cambio_pagina_lado(dat, 'PAG_ADV_IN_PAG_DER', 'PAG_NAC_NEW_PAG_DER')

# UBICACION_PAG_MIX: mitad del catalogo (INICIALES/FINALES), distinto de
# CAMBIO_PAG_LADO (par/impar) -- mismo patron que MODELO_AZZORTI_ROP_NAC_NAC.
dat = calcular_ubicacion_pagina(dat, 'PAG_ADV_IN', 'UBICACION_PAG_ORIGEN')
dat = calcular_ubicacion_pagina(dat, 'PAG_NAC_NEW', 'UBICACION_PAG_DESTINO')
dat['UBICACION_PAG_MIX'] = dat['UBICACION_PAG_ORIGEN'] + '_' + dat['UBICACION_PAG_DESTINO']

# ESTR_MIX: mezcla de estrategia comercial entre origen y destino (mismo
# patron que MODELO_AZZORTI_ROP_NAC_NAC). ESTRATEGIA es la de destino (sin
# sufijo, de cruzar_vehiculo_estrategia en PASO 1); ESTRA_ADV es la de origen
# (renombrada en PASO 2.3 para no chocar con la de destino en este merge).
dat['ESTR_MIX'] = dat['ESTRATEGIA'].astype(str) + '_' + dat['ESTRA_ADV'].astype(str)

# copia intermedia a nivel talla (interim: todavia no es el dataset final de
# modelado). Es el insumo de 0.- MOD_TLL (RUTA_BASE_TALLAS), que la carga tal
# cual para repartir el pronostico por talla -- no rehace esta limpieza
dat.to_parquet(DATA_INTERIM / 'dat.parquet', index=False, compression='snappy')
resumen_filas(dat, 'dat.parquet exportado (nivel talla)')

#%%
#===============================================
# PASO 5: AGRUPAR A NIVEL REFERENCIA (CAMP_NOMBRE_SAP) -- "Modelo Referencias"
#===============================================
titulo_paso('PASO 5: AGRUPAR A NIVEL REFERENCIA (CAMP_NOMBRE_SAP) -- "Modelo Referencias"')

if pais_origen=='COL':
    columnas_suma = ['FAC_NAC', 'FAC_MKT', 'Q_FALT_N', 'Q_MKT_N',
                    'FACT_SAN1', 'FACT_OCC1', 'FACT_ADV_RGN', 'N_SKU']
elif pais_origen=='PER':
    columnas_suma = ['FAC_NAC', 'FAC_MKT', 'Q_FALT_N', 'Q_MKT_N',
                    'FACT_LIMA', 'FACT_COSTA','FACT_SIERRA', 'FACT_SELVA'
                    , 'FACT_ADV_RGN', 'N_SKU']
else :
    raise ValueError("Selección no válida: debe ser 'PER' (Perú) o 'COL' (Colombia).")

columnas_extras = ['CAMP_NOMBRE_SAP', 'PAIS', 'LINEA_FACT', 'TOT_N', 'TOT_A', 'CAMP_CODI_VENT',
                    'PRECIO_N', 'PRECIO_ADV_IN', 'CAMP', 'CAMP_ADV', 'GRUPO', 'SUBGRUPO', 'NOMBRE_SAP',
                    'SUBUEN', 'URL_ADV_IN', 'URL_NAC', 'CATEGORIAS', 'RATIO_PRECIO',
                    'NUM_PRODUCTOS_POR_CATEGORIAS_PAGINA_ORIGEN', 'NUM_PRODUCTOS_POR_CATEGORIAS_ORIGEN',
                    'NUM_PRODUCTOS_POR_SUBGRUPO_ORIGEN', 'PRECIO_PROMEDIO_COMPETIDORES_ORIGEN',
                    'PRECIO_PROMEDIO_SUBGRUPO_ORIGEN', 'RATIO_PROMEDIO_COMPETIDORES_PRECIO_ORIGEN',
                    'RATIO_PROMEDIO_PRECIO_ORIGEN_SUBGRUPO',
                    'NUM_PRODUCTOS_POR_CATEGORIAS_PAGINA_DESTINO', 'NUM_PRODUCTOS_POR_CATEGORIAS_DESTINO',
                    'NUM_PRODUCTOS_POR_SUBGRUPO_DESTINO', 'PRECIO_PROMEDIO_COMPETIDORES_DESTINO',
                    'PRECIO_PROMEDIO_SUBGRUPO_DESTINO', 'RATIO_PROMEDIO_COMPETIDORES_PRECIO_DESTINO',
                    'RATIO_PROMEDIO_PRECIO_DESTINO_SUBGRUPO',
                    'PAG_ADV_IN', 'PAG_NAC_NEW', 'PAG_ADV_IN_PAG_DER', 'PAG_NAC_NEW_PAG_DER', 'CAMBIO_PAG_LADO',
                    'UBICACION_PAG_ORIGEN', 'UBICACION_PAG_DESTINO', 'UBICACION_PAG_MIX', 'ESTR_MIX']

# FACT_BOG/FACT_COS (combinados BOG2+BOG3 / COS1+COS3, ver GRUPOS_FACT_ADV_RGN
# en cargar_datos.py) son exclusivos de COL y no van en columnas_suma, por eso
# se traen aqui como extra. PER no los necesita: sus factores regionales
# (FACT_LIMA/FACT_COSTA/FACT_SIERRA/FACT_SELVA) ya estan en columnas_suma: si
# tambien se agregaran aqui, unir_extras_por_referencia (merge con dat_agg)
# los duplicaria con sufijos _x/_y.
if pais_origen == 'COL':
    columnas_extras += ['FACT_BOG', 'FACT_COS']
#revisa pq  agrupa por ref tiene los factores de regines y en unir extras tiene factores de regiones esta raro
dat_agg = agrupar_por_referencia(dat, columnas_suma)
dat_ref = unir_extras_por_referencia(dat_agg, dat[columnas_extras])
add_track_dest(dat_ref, 'Transformando a por referencia')
# si crecen las filas, alguna referencia trae valores distintos en las columnas extra
resumen_filas(dat_ref, 'columnas extra unidas', antes=dat_agg)
dat_ref = calcular_ratio_factor(dat_ref, col_destino='FAC_NAC', col_origen='FACT_ADV_RGN')

if pais_origen not in FACTORES_REGIONALES_POR_PAIS:
    raise ValueError("Selección no válida: debe ser 'PER' (Perú) o 'COL' (Colombia).")
factores_regionales = FACTORES_REGIONALES_POR_PAIS[pais_origen]

dat_ref = calcular_resumen_factores_regionales(dat_ref, factores_regionales)
dat_ref = marcar_tipo_referencia(dat_ref, col_producto='NOMBRE_SAP', col_camp='CAMP')
dat_ref = calcular_exposicion(dat_ref)
dat_ref = calcular_momentum_subgrupo(dat_ref, ventanas=(3,))

# export para EDA en 2.- notebooks/2.- features.py: dat_ref con RATIO_FACTOR ya
# calculado pero TODAVIA con los posibles atipicos (el filtro de PASO 6 recien
# viene despues), para poder analizarlos ahi tambien -- distinto del CSV final
# de PASO 6 (datos_estimar_5.csv), que ya viene sin atipicos.
dat_ref.to_parquet(DATA_INTERIM / 'dat_ref_5.parquet', index=False, compression='snappy')
resumen_filas(dat_ref, 'dat_ref_5.parquet exportado (con posibles atipicos aun)')

#%%
#===============================================
# PASO 6: FILTRAR POSIBLES ATIPICOS DE RATIO_FACTOR (destino/origen)
#===============================================
titulo_paso('PASO 6: FILTRAR POSIBLES ATIPICOS DE RATIO_FACTOR (destino/origen)')
# marcar_posibles_atipicos_ratio ya imprime su propio '🧹 FILTRO VOLAR FLAG_ATIPICO: N FILAS'
dat_ref = marcar_posibles_atipicos_ratio(dat_ref, col_ratio='RATIO_FACTOR', col_expo='RATIO_EXPOSICION',
                                          nueva_col='FLAG_ATIPICO', volar_atipicos=True)
add_track_dest(dat_ref, 'filtrando ATIPICOS (outliers)')
dat_ref = descomponer_camp(dat_ref, col_camp='CAMP')  # ANIO, IND_CAMP, NUM_CAMP (destino)
dat_ref = descomponer_camp(dat_ref, col_camp='CAMP_ADV', sufix='_ADV')  # ANIO_ADV, IND_CAMP_ADV, NUM_CAMP_ADV (origen)
# NUM_CAMP_MIX: mezcla del numero de campana (1-18) de origen y destino --
# con DESFASE_CAMPANAS fijo casi siempre es el mismo par, salvo en el quiebre
# de anio (campanas 16/17/18 cruzan con 01/02/03 del anio siguiente de forma
# no lineal). Se deja como candidata; si no aporta sobre CAMP_SEN/CAMP_COS,
# Boruta (usar_boruta=True) la descarta.
dat_ref['NUM_CAMP_MIX'] = dat_ref['NUM_CAMP'] + '_' + dat_ref['NUM_CAMP_ADV']
dat_ref.to_csv(DATA_PROCESSED / 'datos_estimar_5.csv', encoding='cp1252', index=False)
resumen_filas(dat_ref, 'datos_estimar_5.csv exportado')

#%%
#===============================================
# PASO 7: SEGUNDA PASADA DE INFLADOS (origen, TIPO_ZONA='A')
#===============================================
titulo_paso("PASO 7: SEGUNDA PASADA DE INFLADOS (origen, TIPO_ZONA='A')")
df_inflados_origen = load_inflados(pais=pais_origen, tipo_zona='A')
df_inflados_origen = calcular_camp_mas_desfase(df_inflados_origen, col_camp='CAMP', desfase_campanas=desfase_campanas)
df_inflados_origen['CAMP_NOMBRE_SAP'] = (df_inflados_origen[f'CAMP_MAS_{desfase_campanas}'].astype(str) + '_'
                                          + df_inflados_origen['NOMBRE_SAP'].astype(str))
df_inflados_origen = df_inflados_origen.drop_duplicates(['CAMP_NOMBRE_SAP'], keep='first')

n_antes = len(dat_ref)
dat_ref = pd.merge(dat_ref, df_inflados_origen[['PROB_ATIPICO', 'CAMP_NOMBRE_SAP']], on='CAMP_NOMBRE_SAP', how='left')
dat_ref['PROB_ATIPICO'] = dat_ref['PROB_ATIPICO'].apply(lambda x: 'NO' if pd.isna(x) else 'SI')
dat_ref = dat_ref[dat_ref['PROB_ATIPICO'] == 'NO'].copy()
add_track_dest(dat_ref, f'volar infladas origen ({pais_origen}, TIPO_ZONA=A)')
resumen_filas(dat_ref, f'volar infladas origen ({pais_origen}, TIPO_ZONA=A)', antes=n_antes)

# -------------------------------------------------------------
# Exportar la bitacora de filtros a excel
excel_dir = Path(__file__).resolve().parents[2] / '5.- reports' / 'excel'
excel_dir.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(excel_dir / 'FILTROS.xlsx') as writer:
    pd.DataFrame(bitacora_destino).to_excel(writer, sheet_name=f'DESTINO_{pais_destino}', index=False)
    pd.DataFrame(bitacora_origen).to_excel(writer, sheet_name=f'ORIGEN_{pais_origen}', index=False)
print(f"✅ Excel de FILTROS exportado en 5.- reports/excel/FILTROS.xlsx")
# -------------------------------------------------------------

#%%
#===============================================
# PASO 8: SPLIT TRAIN/TEST POR CORTE DE CAMPANA
#===============================================
titulo_paso('PASO 8: SPLIT TRAIN/TEST POR CORTE DE CAMPANA')
train, test = dividir_train_test_por_corte(dat_ref, col_camp='CAMP', corte=corte_train_test)

datos_finales = pd.concat([train, test], ignore_index=True)
datos_finales['NUM_CAMP_INT'] = datos_finales['IND_CAMP'].str[-2:].astype(int)
datos_finales = calcular_variables_ciclicas_campana(datos_finales, col_num_camp='NUM_CAMP_INT', n_camps=18)
datos_finales['RATIO_PAG_NAC_ADV'] = datos_finales['PAG_NAC_NEW'] / datos_finales['PAG_ADV_IN']

#%%
#===============================================
# PASO 9: ARMAR X/Y PARA OPTUNA
#===============================================
titulo_paso('PASO 9: ARMAR X/Y PARA OPTUNA')
datos_finales['CAMP'] = datos_finales['CAMP'].astype(int)
datos_finales['CLASIF'] = np.where(datos_finales['CAMP'] >= corte_train_test, 'TEST', 'TRAIN')
# alias con el mismo nombre que MODELO_AZZORTI_ROP_NAC_NAC (FACTOR_RCP_ORIGEN/
# FACTOR_RCP_DESTINO) para poder reusar sus funciones de evaluacion.py sin
# modificarlas -- reemplaza a FACTOR_NAL/Y_STAR
datos_finales['FACTOR_RCP_DESTINO'] = datos_finales['FAC_NAC']
datos_finales['FACTOR_RCP_ORIGEN'] = datos_finales['FACT_ADV_RGN']
# target del modelo, en escala log (igual que SBR_SVR_RF.py: LOG_RATIO =
# log(RATIO), sin el log-shift +0.0001*1000 que usa NAC_NAC) -- invert_y con
# y_mode='log_ratio' asume justamente esta escala (exp(pred)*ratio_col)
datos_finales['LOG_RATIO_FACTOR'] = np.log(datos_finales['RATIO_FACTOR'])

# export para 6.- experimentos_nube/ (RF NUBE.py espera la columna SPLIT con
# TRAIN/TEST -- mismo alias que CLASIF, ver README de experimentos_nube)
datos_finales['SPLIT'] = datos_finales['CLASIF']
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
datos_finales.to_parquet(DATA_PROCESSED / 'df_limpio.parquet', index=False, compression='snappy')
resumen_filas(datos_finales, 'df_limpio.parquet exportado (train+test)')

train = datos_finales[datos_finales['CLASIF'] == 'TRAIN']
valid = datos_finales[datos_finales['CLASIF'] == 'TEST']
log(f'TRAIN: {len(train):,} filas (campañas < {corte_train_test}) | TEST: {len(valid):,} filas (campañas >= {corte_train_test})')

# que set de variables se usa:
#   usar_boruta=True  -> set completo de candidatas (obtener_features_candidatas,
#                        en construir_features.py) y Boruta lo reduce abajo
#   usar_boruta=False -> FEATURES_FIJAS (arriba, junto a los flags)

# se copian las listas para que Boruta no modifique el original
set_base = obtener_features_candidatas(factores_regionales) if usar_boruta else FEATURES_FIJAS
features = {
    'target': set_base['target'],
    'num': list(set_base['num']),
    'cat': list(set_base['cat']),
}
log(f"Features: {'CANDIDATAS (Boruta)' if usar_boruta else 'FIJAS (a mano)'} -> "
    f"{len(features['num'])} num + {len(features['cat'])} cat")
features['all'] = features['num'] + features['cat']

X_train, y_train = armar_x_y(train, features)
X_test, y_test = armar_x_y(valid, features)
log(f'X_train: {X_train.shape[0]:,} filas x {X_train.shape[1]} variables | '
    f'X_test: {X_test.shape[0]:,} filas x {X_test.shape[1]} variables')

if usar_boruta:
    features = seleccionar_features_boruta(X_train, y_train, features, max_iter=max_iter, seed=seed)
    X_train, y_train = armar_x_y(train, features)
    X_test, y_test = armar_x_y(valid, features)
    log(f'Tras Boruta quedan {len(features["all"])} variables')

#%%
#===============================================
# PASO 9.1: ALERTAS DE DATA DRIFT ENTRE TRAIN Y TEST
#===============================================
titulo_paso('PASO 9.1: ALERTAS DE DATA DRIFT ENTRE TRAIN Y TEST')
from models.data_drift import calcular_mmd, reporte_psi
from sklearn.preprocessing import StandardScaler

cols_drift = [c for c in features['num'] if c in X_train.columns]
if len(cols_drift) > 0:
    # 1. PSI (Univariado)
    df_psi = reporte_psi(X_train, X_test, cols_drift)
    variables_en_peligro = df_psi[df_psi['Estado'] == 'PELIGRO']
    if not variables_en_peligro.empty:
        log("⚠️ ALERTA DE DRIFT UNIVARIADO (PSI > 0.2) en las siguientes variables:")
        for _, row in variables_en_peligro.iterrows():
            log(f"   - {row['Variable']}: {row['PSI']}")
    else:
        log("✅ Sin alertas graves de drift univariado.")
        
    # 2. MMD (Multivariado)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_num = imputer.fit_transform(X_train[cols_drift])
    X_train_mmd = scaler.fit_transform(X_train_num)
    
    X_test_num = imputer.transform(X_test[cols_drift])
    X_test_mmd = scaler.transform(X_test_num)
    
    res_mmd = calcular_mmd(X_train_mmd, X_test_mmd, n_perm=50)
    mmd_score = res_mmd['mmd_score']
    p_value = res_mmd['p_value']
    
    log(f"📊 Puntaje MMD Multivariado: {mmd_score:.4f} (p-value: {p_value:.4f})")
    if p_value < 0.05:
        log("⚠️ ALERTA DE DRIFT MULTIVARIADO: La estructura general de los datos de Test es muy distinta a la de Train (p < 0.05).")
else:
    log("⚠️ No hay variables numéricas disponibles para calcular Drift.")

#%%
#===============================================
# PASO 10: OPTUNA (mismo enfoque simple que MODELO_AZZORTI_ROP_NAC_NAC: RF +
# CatBoostEncoder en un Pipeline, evaluado con R2 sobre TEST)
#===============================================
titulo_paso('PASO 10: OPTUNA (RF + CatBoostEncoder en Pipeline, evaluado con R2 sobre TEST)')
if usar_optuna:
    camp_train = train['CAMP'] if usar_ponderacion_temporal else None
    objetivo = construir_objetivo_optuna(X_train, y_train, X_test, y_test, features, seed=seed, train_camp=camp_train)
    best_hp = ejecutar_estudio_optuna(objetivo, n_trials=n_trials, seed=seed)
else:
    # mejor trial encontrado en una corrida de prueba (20 trials, ECU/RIN) --
    # actualizar aca despues de correr con usar_optuna=True y n_trials=500
    best_hp = {'n_estimators': 1059, 'max_depth': 14, 'min_samples_leaf': 4,
               'min_samples_split': 17, 'max_features': 0.46003695381676873}
print('Mejores features:', features['all'])
print('Mejores hiperparametros:', best_hp)

#%%
#===============================================
# PASO 11: FIT DEL PIPELINE FINAL + GUARDAR .pkl
#===============================================
titulo_paso('PASO 11: FIT DEL PIPELINE FINAL + GUARDAR .pkl')
log(f'Entrenando pipeline con {X_train.shape[0]:,} filas y {X_train.shape[1]} variables '
    f'(hiperparametros {"de Optuna" if usar_optuna else "fijos"})...')

if usar_ponderacion_temporal and 'decay_rate' in best_hp:
    camp_str = train['CAMP'].astype(str)
    anios = camp_str.str[:4].astype(int)
    ciclos = camp_str.str[4:].astype(int)
    tiempo_absoluto = anios * 18 + ciclos
    distancia = tiempo_absoluto.max() - tiempo_absoluto
    sample_weights = np.exp(-best_hp['decay_rate'] * distancia)
    
    # Remover decay_rate de best_hp porque RandomForest no lo acepta como parámetro nativo
    hp_modelo = {k: v for k, v in best_hp.items() if k != 'decay_rate'}
    pipe = construir_pipeline(features, hp_modelo, seed=seed)
    
    pipe.fit(X_train, y_train, modelo__sample_weight=sample_weights)
else:
    # Remover decay_rate por precaucion si estuviera en best_hp
    hp_modelo = {k: v for k, v in best_hp.items() if k != 'decay_rate'}
    pipe = construir_pipeline(features, hp_modelo, seed=seed)
    pipe.fit(X_train, y_train)

log('Pipeline entrenado')

if guardar_pkl:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(pipe, MODELS_DIR / nombre_modelo)
    print(f'Pipeline guardada en {nombre_modelo}')

#%%
#===============================================
# PASO 12: PREDICCIONES POR ARBOL Y PERCENTILES (train/test)
#===============================================
titulo_paso('PASO 12: PREDICCIONES POR ARBOL Y PERCENTILES (train/test)')
predicciones_test = predicciones_por_arbol(pipe, X_test, base_df=valid, ratio_col='FACTOR_RCP_ORIGEN')
percentiles_test = calcular_percentiles(predicciones_test)

predicciones_train = predicciones_por_arbol(pipe, X_train, base_df=train, ratio_col='FACTOR_RCP_ORIGEN')
percentiles_train = calcular_percentiles(predicciones_train)
log(f'Percentiles por arbol calculados ({predicciones_test.shape[1]} arboles) para train y test')

#%%
#===============================================
# PASO 13: METRICAS EN ESCALA REAL (FACTOR_RCP_DESTINO)
#===============================================
titulo_paso('PASO 13: METRICAS EN ESCALA REAL (FACTOR_RCP_DESTINO)')
df_train_pred = agregar_percentiles_y_error(train, percentiles_train)
df_test_pred = agregar_percentiles_y_error(valid, percentiles_test)

metricas('TRAIN', df_train_pred['FACTOR_RCP_DESTINO'], df_train_pred['Q_50'])
metricas('TEST', df_test_pred['FACTOR_RCP_DESTINO'], df_test_pred['Q_50'])

svs_train, sbr_train = svs_sbr(df_train_pred['Q_50'], df_train_pred['FACTOR_RCP_DESTINO'])
svs_test, sbr_test = svs_sbr(df_test_pred['Q_50'], df_test_pred['FACTOR_RCP_DESTINO'])
print(f'TRAIN -> SVS: {svs_train}% SBR: {sbr_train}%')
print(f'TEST  -> SVS: {svs_test}% SBR: {sbr_test}%')

#%%
#===============================================
# PASO 14: IMPORTANCIA DE VARIABLES
#===============================================
titulo_paso('PASO 14: IMPORTANCIA DE VARIABLES')
importancia = obtener_importancia_variables(pipe, features)
print(importancia)

#%%
#===============================================
# PASO 15: CONSOLIDAR RESULTADOS Y EXPORTAR A "PRUEBAS DE MODELO"
# Un csv por corrida (mismo esquema que MODELO_AZZORTI_ROP_NAC_NAC); para
# compararlas juntas correr PRUEBAS DE MODELO/consolidar_pruebas_modelo.py
#===============================================
titulo_paso('PASO 15: CONSOLIDAR RESULTADOS Y EXPORTAR A "PRUEBAS DE MODELO"')
df_final = consolidar_resultados(df_train_pred, df_test_pred, comentario)
df_final['VARIABLES'] = str(features['all'])
# NIVEL separa en el consolidado/Power BI las corridas de referencia de las de talla (0.- MOD_TLL)
df_final['NIVEL'] = 'REFERENCIA'

PRUEBAS_MODELO_DIR.mkdir(parents=True, exist_ok=True)
nombre_archivo_pruebas = f'df_{pais_origen}_{pais_destino}_{linea}_{comentario}.csv'
df_final.to_csv(PRUEBAS_MODELO_DIR / nombre_archivo_pruebas, index=False, encoding='cp1252')
resumen_filas(df_final, f'exportado PRUEBAS_MODELO/{nombre_archivo_pruebas}')

# NUEVO: Conteo de Referencias por Campaña para Auditoría (DSB)
df_auditoria = df_final.groupby('CAMP')['NOMBRE_SAP'].nunique().reset_index()
df_auditoria.rename(columns={'NOMBRE_SAP': 'CANTIDAD_REFERENCIAS'}, inplace=True)
nombre_archivo_auditoria = f'Auditoria_Cantidades_{pais_origen}_{pais_destino}_{linea}_{comentario}.xlsx'
excel_dir = Path(__file__).resolve().parents[2] / '5.- reports' / 'excel'
excel_dir.mkdir(parents=True, exist_ok=True)
df_auditoria.to_excel(excel_dir / nombre_archivo_auditoria, index=False)
log(f'✅ Auditoría de cantidades (referencias por campaña) exportada en 5.- reports/excel/{nombre_archivo_auditoria}')

# copia en las pruebas de 0.- MOD_TLL (alli se consolida referencia vs talla para el Power BI)
if TLL_ROOT.exists():
    PRUEBAS_MODELO_TLL_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(PRUEBAS_MODELO_TLL_DIR / nombre_archivo_pruebas, index=False, encoding='cp1252')
    resumen_filas(df_final, f'copia en PRUEBAS_MODELO de 0.- MOD_TLL/{nombre_archivo_pruebas}')

resultado_test_por_anio = svs_sbr_por_grupo(df_test_pred, 'ANIO', col_pred='Q_50', col_real='FACTOR_RCP_DESTINO')
print('TEST (por anio):')
print(resultado_test_por_anio)

#%%
#===============================================
# PASO 16: SVS/SBR POR BIN DE Q70_Q30 (mismo analisis que MODELO_AZZORTI_ROP_NAC_NAC)
#===============================================
titulo_paso('PASO 16: SVS/SBR POR BIN DE Q70_Q30 (mismo analisis que MODELO_AZZORTI_ROP_NAC_NAC)')
resultado_largo = analizar_svs_sbr_por_bin_ratio(df_final, col_ratio='Q70_Q30', corte_fijos=False)
graficar_svs_sbr_por_bin(resultado_largo)
plt.show()

# TODO: falta predecir.py (aplicar este .pkl sobre campanas nuevas, no solo
# sobre el split train/test de esta corrida) -- ver README.md de este modelo.
