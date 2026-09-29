"""
Orquesta el entrenamiento del MODELO POR TALLAS de ROP_ADV_NAC: misma
estructura de pasos y flags que 0.- MOD_RF/3.- src/models/entrenar.py, pero
a nivel talla (CODI_VENT) en vez de referencia (CAMP_NOMBRE_SAP).

Migrado de los scripts ECU_ADV_NAC_RIN_MODELO_TALLAS.py y
BOL_ADV_NAC_REX_MODELO_TALLAS.py:
- Predice la participacion de cada talla dentro de su referencia en el NAC
  (PART_NAC) a partir de su participacion en el ADV (PART_ADV) y del volumen
  ADV de la referencia.
- Rebalancea para que las tallas de cada referencia sumen 1 y reparte los
  percentiles (LIM_INF..Q_90..LIM_SUP) del pronostico de MOD_RF.
- Compara SVS/SBR del modelo de referencia vs el de tallas (PASO 13).

Requisito: correr antes 0.- MOD_RF/3.- src/models/entrenar.py, que exporta
los dos insumos que aqui SOLO se cargan (ver config.py):
- RUTA_BASE_TALLAS: base por talla ya cruzada ADV/NAC (interim/dat.parquet)
- RUTA_PRED_REFERENCIA: pronostico final por referencia (csv de PRUEBAS DE MODELO)
"""
#%%
import sys
from pathlib import Path  # manipular rutas de forma OOP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (PAIS_DESTINO, PAIS_ORIGEN, LINEA, NOMBRE_PKL, MODELS_DIR, PRUEBAS_MODELO_DIR,
                    DATA_PROCESSED, REPORTS_EXCEL, REPORTS_FIGURES,
                    RUTA_BASE_TALLAS, RUTA_PRED_REFERENCIA, CORTE_TRAIN_TEST, COMENTARIO_REG)
from data.mis_funciones import descomponer_camp, metricas, svs_sbr, svs_sbr_por_grupo, log, resumen_filas, titulo_paso
from features.construir_features import FEATURES_CANDIDATAS_TALLAS, calcular_participacion_talla
from models.entrenamiento_utils import (dividir_train_test_por_corte, armar_x_y,
                                         seleccionar_features_boruta, construir_objetivo_optuna,
                                         ejecutar_estudio_optuna)
from models.pipeline import construir_pipeline
from models.evaluacion import (obtener_importancia_variables, agregar_percentiles_y_error,
                                consolidar_resultados, analizar_svs_sbr_por_bin_ratio,
                                rebalancear_participacion, repartir_percentiles_referencia)
from visualization.graficos import graficar_svs_sbr_por_bin

# paises, corte y comentario vienen de config_proyecto.py (raiz del
# proyecto), los mismos que usa 0.- MOD_RF: NO editarlos aqui. La ventana de
# campañas ya viene aplicada en la base que exporta MOD_RF
pais_destino = PAIS_DESTINO
pais_origen = PAIS_ORIGEN
linea = LINEA

corte_train_test = CORTE_TRAIN_TEST
seed = 42


usar_boruta = False  # True para reducir variables con Boruta antes de Optuna
max_iter = 100        # iteraciones de Boruta: bajarlo reduce el tiempo, a costa de una seleccion menos estable

usar_optuna = True  # False usa los hiperparametros fijos de abajo (mas rapido); True corre Optuna
n_trials = 10        # bajarlo reduce el tiempo de corrida, a costa de una busqueda menos fina


comentario = f'{COMENTARIO_REG} TALLAS'
nombre_modelo = NOMBRE_PKL
guardar_pkl = True   # True reemplaza el .pkl anterior en 4.- models/

# Set FIJO de variables (se usa si usar_boruta=False). Hoy es igual a
# FEATURES_CANDIDATAS_TALLAS (las del script BOL_ADV_NAC_REX); editar aqui al
# revisar el importance.
FEATURES_FIJAS = {
    'target': 'PART_NAC',
    'num': ['PART_ADV', 'SUM_FACT_ADV', 'FACT_ADV_RGN', 'NUM_TALLAS'],
    'cat': ['SUBGRUPO', 'TALLA_NEW'],
}

#%%
#===============================================
# PASO 1: CARGAR BASE POR TALLA DE 0.- MOD_RF
# Ya limpia y cruzada ADV/NAC (MOD_RF PASO 1-4); aqui no se rehace
#===============================================
titulo_paso('PASO 1: CARGAR BASE POR TALLA DE 0.- MOD_RF')
dat_tallas = pd.read_parquet(RUTA_BASE_TALLAS)
resumen_filas(dat_tallas, f'base por talla leida ({RUTA_BASE_TALLAS.name})')

# el cruce origen/destino de MOD_RF deja CODI_VENT_x (destino) / CODI_VENT_y (origen)
dat_tallas = dat_tallas.rename(columns={'CODI_VENT_x': 'CODI_VENT'})
dat_tallas['CAMP'] = dat_tallas['CAMP'].astype(int)

#%%
#===============================================
# PASO 2: PARTICIPACION POR TALLA (NAC y ADV)
#===============================================
titulo_paso('PASO 2: PARTICIPACION POR TALLA (NAC y ADV)')
dat_tallas = calcular_participacion_talla(dat_tallas)

# alias con el mismo nombre que MOD_RF (FACTOR_RCP_DESTINO) para reusar
# evaluacion.py sin modificarla: el real de cada talla
dat_tallas['FACTOR_RCP_DESTINO'] = dat_tallas['FAC_NAC']
dat_tallas = descomponer_camp(dat_tallas, col_camp='CAMP')  # ANIO para el SVS/SBR por año

#%%
#===============================================
# PASO 3: SPLIT TRAIN/TEST POR CORTE DE CAMPANA (mismo corte que MOD_RF)
#===============================================
titulo_paso('PASO 3: SPLIT TRAIN/TEST POR CORTE DE CAMPANA')
train, valid = dividir_train_test_por_corte(dat_tallas, col_camp='CAMP', corte=corte_train_test)
train['CLASIF'] = 'TRAIN'
valid['CLASIF'] = 'TEST'
datos_finales = pd.concat([train, valid], ignore_index=True)
datos_finales['SPLIT'] = datos_finales['CLASIF']
train = datos_finales[datos_finales['CLASIF'] == 'TRAIN'].copy()
valid = datos_finales[datos_finales['CLASIF'] == 'TEST'].copy()

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
datos_finales.to_parquet(DATA_PROCESSED / 'df_limpio.parquet', index=False, compression='snappy')
resumen_filas(datos_finales, 'df_limpio.parquet exportado (train+test)')
log(f'TRAIN: {len(train):,} filas (campañas < {corte_train_test}) | TEST: {len(valid):,} filas (campañas >= {corte_train_test})')

#%%
#===============================================
# PASO 4: ARMAR X/Y (y Boruta opcional)
#===============================================
titulo_paso('PASO 4: ARMAR X/Y')
# que set de variables se usa:
#   usar_boruta=True  -> FEATURES_CANDIDATAS_TALLAS (en construir_features.py) y Boruta lo reduce
#   usar_boruta=False -> FEATURES_FIJAS (arriba, junto a los flags)
set_base = FEATURES_CANDIDATAS_TALLAS if usar_boruta else FEATURES_FIJAS
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
# PASO 5: HIPERPARAMETROS (Optuna u fijos)
#===============================================
titulo_paso('PASO 5: HIPERPARAMETROS')
if usar_optuna:
    objetivo = construir_objetivo_optuna(X_train, y_train, X_test, y_test, features, seed=seed)
    best_hp = ejecutar_estudio_optuna(objetivo, n_trials=n_trials, seed=seed)
else:
    # punto de partida: los del modelo de tallas ECU_ADV_NAC_RIN (Optuna, 50 trials)
    best_hp = {'n_estimators': 526, 'max_depth': 23, 'min_samples_leaf': 10,
               'min_samples_split': 12, 'max_features': 1.0}
print('Mejores features:', features['all'])
print('Mejores hiperparametros:', best_hp)

#%%
#===============================================
# PASO 6: FIT DEL PIPELINE FINAL + GUARDAR .pkl
#===============================================
titulo_paso('PASO 6: FIT DEL PIPELINE FINAL + GUARDAR .pkl')
pipe = construir_pipeline(features, best_hp, seed=seed)
log(f'Entrenando pipeline con {X_train.shape[0]:,} filas y {X_train.shape[1]} variables '
    f'(hiperparametros {"de Optuna" if usar_optuna else "fijos"})...')
pipe.fit(X_train, y_train)
log('Pipeline entrenado')

if guardar_pkl:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(pipe, MODELS_DIR / nombre_modelo)
    print(f'Pipeline guardada en {nombre_modelo}')

#%%
#===============================================
# PASO 7: PREDECIR Y REBALANCEAR PARTICIPACION
# PART_P_50 = modelo; PART_P_BASE = baseline sin modelo (reparto igual al ADV)
#===============================================
titulo_paso('PASO 7: PREDECIR Y REBALANCEAR PARTICIPACION')
train['PRED_PART'] = pipe.predict(X_train)
valid['PRED_PART'] = pipe.predict(X_test)

metricas('TRAIN (participacion)', y_train, train['PRED_PART'])
metricas('TEST (participacion)', y_test, valid['PRED_PART'])

train = rebalancear_participacion(train, col_pred='PRED_PART', nueva_col='PART_P_50')
valid = rebalancear_participacion(valid, col_pred='PRED_PART', nueva_col='PART_P_50')
train = rebalancear_participacion(train, col_pred='PART_ADV', nueva_col='PART_P_BASE')
valid = rebalancear_participacion(valid, col_pred='PART_ADV', nueva_col='PART_P_BASE')

assert np.allclose(valid.groupby('CAMP_NOMBRE_SAP')['PART_P_50'].sum(), 1), 'PART_P_50 no suma 1 por referencia'

#%%
#===============================================
# PASO 8: REPARTIR PERCENTILES DE MOD_RF
# Pronostico por referencia (df_final de MOD_RF) x participacion rebalanceada
#===============================================
titulo_paso('PASO 8: REPARTIR PERCENTILES DE MOD_RF')
df_ref = pd.read_csv(RUTA_PRED_REFERENCIA, encoding='cp1252').rename(columns={'y_pred': 'Q_50'})
resumen_filas(df_ref, f'pronostico de referencia leido ({RUTA_PRED_REFERENCIA.name})')
n_dup = df_ref.duplicated(subset=['CAMP', 'NOMBRE_SAP']).sum()
if n_dup > 0:
    print(f'⚠️ AVISO: {n_dup} CAMP+NOMBRE_SAP duplicados en el pronostico de referencia')

train_cruz, percentiles_train = repartir_percentiles_referencia(train, df_ref)
valid_cruz, percentiles_test = repartir_percentiles_referencia(valid, df_ref)

df_train_pred = agregar_percentiles_y_error(train_cruz, percentiles_train)
df_test_pred = agregar_percentiles_y_error(valid_cruz, percentiles_test)

# la suma de las tallas debe devolver exactamente el pronostico de la referencia
suma_tallas = df_test_pred.groupby('CAMP_NOMBRE_SAP').agg(Q_50=('Q_50', 'sum'), REF_Q_50=('REF_Q_50', 'first'))
assert np.allclose(suma_tallas['Q_50'], suma_tallas['REF_Q_50']), 'la suma de Q_50 por talla no cuadra con la referencia'

#%%
#===============================================
# PASO 9: METRICAS EN ESCALA REAL (unidades por talla)
#===============================================
titulo_paso('PASO 9: METRICAS EN ESCALA REAL (unidades por talla)')
metricas('TRAIN', df_train_pred['FACTOR_RCP_DESTINO'], df_train_pred['Q_50'])
metricas('TEST', df_test_pred['FACTOR_RCP_DESTINO'], df_test_pred['Q_50'])

svs_train, sbr_train = svs_sbr(df_train_pred['Q_50'], df_train_pred['FACTOR_RCP_DESTINO'])
svs_test, sbr_test = svs_sbr(df_test_pred['Q_50'], df_test_pred['FACTOR_RCP_DESTINO'])
print(f'TRAIN -> SVS: {svs_train}% SBR: {sbr_train}%  (optimista: percentiles de MOD_RF dentro de muestra)')
print(f'TEST  -> SVS: {svs_test}% SBR: {sbr_test}%')

#%%
#===============================================
# PASO 10: IMPORTANCIA DE VARIABLES
#===============================================
titulo_paso('PASO 10: IMPORTANCIA DE VARIABLES')
importancia = obtener_importancia_variables(pipe, features)
print(importancia)

#%%
#===============================================
# PASO 11: CONSOLIDAR RESULTADOS Y EXPORTAR A "PRUEBAS DE MODELO"
#===============================================
titulo_paso('PASO 11: CONSOLIDAR RESULTADOS Y EXPORTAR A "PRUEBAS DE MODELO"')
df_final = consolidar_resultados(df_train_pred, df_test_pred, comentario)
df_final['VARIABLES'] = str(features['all'])
# NIVEL separa en el consolidado/Power BI las corridas de talla de las de referencia (0.- MOD_RF)
df_final['NIVEL'] = 'TALLA'

PRUEBAS_MODELO_DIR.mkdir(parents=True, exist_ok=True)
nombre_archivo_pruebas = f'df_{pais_origen}_{pais_destino}_{linea}_{comentario}.csv'
df_final.to_csv(PRUEBAS_MODELO_DIR / nombre_archivo_pruebas, index=False, encoding='cp1252')
resumen_filas(df_final, f'exportado PRUEBAS_MODELO/{nombre_archivo_pruebas}')

resultado_test_por_anio = svs_sbr_por_grupo(df_test_pred, 'ANIO', col_pred='Q_50', col_real='FACTOR_RCP_DESTINO')
print('TEST (por anio):')
print(resultado_test_por_anio)

#%%
#===============================================
# PASO 12: SVS/SBR POR BIN DE Q70_Q30 (mismo analisis que MOD_RF)
#===============================================
titulo_paso('PASO 12: SVS/SBR POR BIN DE Q70_Q30')
resultado_largo = analizar_svs_sbr_por_bin_ratio(df_final, col_ratio='Q70_Q30', corte_fijos=False)
graficar_svs_sbr_por_bin(resultado_largo)
plt.show()

#%%
#===============================================
# PASO 13: COMPARACION REFERENCIA VS TALLAS (SVS/SBR en TEST)
# Cuanto SVS se pierde / SBR se gana al bajar el pronostico de referencia a
# talla, y si el RF le gana a repartir igual que el ADV (baseline sin modelo).
# Mismo universo de referencias en los 3 bloques (TEST, inner con MOD_RF).
#===============================================
titulo_paso('PASO 13: COMPARACION REFERENCIA VS TALLAS')
cols_base = ['CAMP_NOMBRE_SAP', 'ANIO']

# REFERENCIA: una fila por CAMP_NOMBRE_SAP con el pronostico/real de MOD_RF
df_cmp_ref = (df_test_pred.drop_duplicates('CAMP_NOMBRE_SAP')[cols_base + ['REF_Q_50', 'REF_y_real']]
              .rename(columns={'REF_Q_50': 'Q_PRED', 'REF_y_real': 'Q_REAL'})
              .assign(MODELO='REFERENCIA'))

# TALLAS_RF: una fila por talla con el Q_50 repartido por el modelo
df_cmp_rf = (df_test_pred[cols_base + ['Q_50', 'FACTOR_RCP_DESTINO']]
             .rename(columns={'Q_50': 'Q_PRED', 'FACTOR_RCP_DESTINO': 'Q_REAL'})
             .assign(MODELO='TALLAS_RF'))

# TALLAS_BASELINE: mismo reparto pero con la participacion del ADV, sin modelo
df_cmp_base = (df_test_pred[cols_base + ['FACTOR_RCP_DESTINO']]
               .assign(Q_PRED=df_test_pred['PART_P_BASE'] * df_test_pred['REF_Q_50'], MODELO='TALLAS_BASELINE')
               .rename(columns={'FACTOR_RCP_DESTINO': 'Q_REAL'}))

df_comparacion = pd.concat([df_cmp_ref, df_cmp_rf, df_cmp_base], ignore_index=True)
resumen_filas(df_comparacion, 'base larga de comparacion')

# control: el real de la referencia debe ser la suma del real de sus tallas
dif_real = df_cmp_rf.groupby('CAMP_NOMBRE_SAP')['Q_REAL'].sum() - df_cmp_ref.set_index('CAMP_NOMBRE_SAP')['Q_REAL']
log(f'Diferencia max. real referencia vs suma de tallas: {dif_real.abs().max():.4f}')

orden_modelos = ['REFERENCIA', 'TALLAS_RF', 'TALLAS_BASELINE']
svs_sbr_modelos = svs_sbr_por_grupo(df_comparacion, 'MODELO', col_pred='Q_PRED', col_real='Q_REAL')
svs_sbr_modelos = svs_sbr_modelos.set_index('MODELO').loc[orden_modelos].reset_index()
print('TEST (total):')
print(svs_sbr_modelos.to_string(index=False))

svs_sbr_modelos_anio = svs_sbr_por_grupo(df_comparacion, ['ANIO', 'MODELO'], col_pred='Q_PRED', col_real='Q_REAL')
print('TEST (por año):')
print(svs_sbr_modelos_anio.to_string(index=False))

# grafico: reutiliza graficar_svs_sbr_por_bin con MODELO en el eje x (columna RATIO)
comparacion_largo = (svs_sbr_modelos.rename(columns={'MODELO': 'RATIO'})
                     .melt(id_vars='RATIO', value_vars=['SVS', 'SBR'], var_name='METRICA', value_name='VALOR'))
ax = graficar_svs_sbr_por_bin(comparacion_largo, orden=orden_modelos)
ax.set(xlabel='Modelo', ylabel='% sobre el real',
       title=f'SVS / SBR en test: referencia vs tallas ({pais_origen}->{pais_destino} {linea})')
REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)
plt.savefig(REPORTS_FIGURES / 'comparacion_ref_vs_tallas.png', bbox_inches='tight', dpi=120)
plt.show()

REPORTS_EXCEL.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(REPORTS_EXCEL / f'comparacion_ref_vs_tallas_{comentario}.xlsx') as writer:
    svs_sbr_modelos.to_excel(writer, sheet_name='TOTAL', index=False)
    svs_sbr_modelos_anio.to_excel(writer, sheet_name='POR_ANIO', index=False)
log(f'Comparacion exportada a {REPORTS_EXCEL.name} y {REPORTS_FIGURES.name}')

# Lectura: el SVS por talla siempre queda <= al de referencia (suma el error
# de reparto al de volumen); la brecha REFERENCIA - TALLAS_RF es el "costo"
# de bajar a talla. Si TALLAS_RF no le gana a TALLAS_BASELINE, el RF no se justifica.

# TODO: falta predecir.py (aplicar este .pkl sobre campanas nuevas), igual que en MOD_RF.
