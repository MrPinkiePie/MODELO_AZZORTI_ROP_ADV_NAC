#%%

# -*- coding: utf-8 -*-
"""
Busqueda combinatoria de variables para el Random Forest de ROP_ADV_NAC.
Adaptado de MODELO_AZZORTI_ROP_NAC_NAC/6.- experimentos_nube/RF NUBE.py: misma
logica, pero con la lista de features real de este modelo (ver 'features' en
3.- src/models/entrenar.py) y agrupando por 'CAMP' (ADV_NAC no tiene columna
CAMP_DESTINO, la campana destino es simplemente CAMP).
"""

from itertools import combinations, chain
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from category_encoders import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import os
from pathlib import Path  # resolver la carpeta del archivo para el export a excel

import warnings
warnings.filterwarnings("ignore")

# Este script vive en 0.- MOD_RF/6.- experimentos_nube/, por eso parents[1] es 0.- MOD_RF
DEV_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTOS_NUBE = Path(__file__).resolve().parent

comentario = "prueba_1"
USAR_TOP_K_FEATURES = False  # False = usa todas las numeric_features+categorical_features (busqueda completa)
N_ESTIMATORS_BUSQUEDA = 100  # arboles del RF SOLO durante la busqueda combinatoria; subir para el modelo final
TOP_K_FEATURES = 11

# ----1. Cargar los datos ya armados por entrenar.py (PASO 9: export a df_limpio.parquet)
df = pd.read_parquet(DEV_ROOT / '0.- data' / 'processed' / 'df_limpio.parquet')

TRAIN = df[df["SPLIT"] == "TRAIN"].copy()
TEST = df[df["SPLIT"].isin(["TEST", 'VALIDACION'])].copy()

print("Train shape:", TRAIN.shape)
print("Test shape:", TEST.shape)

# --- CONFIGURACION TARGET ---
TARGET_SELECCIONADO = 'RATIO_FACTOR' # Puede ser 'LOG_RATIO_FACTOR' o 'RATIO_FACTOR'
target = [TARGET_SELECCIONADO]

cycle_features = []

# Candidatas numericas/categoricas -- mismo listado que 'features' en
# entrenar.py, sin los factores regionales crudos (FACT_BOG/FACT_LIMA/etc,
# distintos segun PAIS_ORIGEN COL/PER): esos ya quedan resumidos en
# FACTOR_MAX/MIN/MEDIANA, que si son comunes a ambos paises. Si se quiere
# probar tambien los crudos, agregarlos aca a mano segun el PAIS_ORIGEN usado
# en la corrida de entrenar.py que genero df_limpio.parquet.
numeric_features = [
    'RATIO_PRECIO', 'FACT_ADV_RGN', 'FACTOR_MAX', 'FACTOR_MIN', 'FACTOR_MEDIANA', 
    'PRECIO_PROMEDIO_COMPETIDORES_ORIGEN', 'PRECIO_PROMEDIO_SUBGRUPO_ORIGEN',
    'MOMENTUM_SUBGRUPO_LAG_3', 'HHI_CLIMA_ORIGEN', 'MOMENTUM_CLIMA_LAG_3'
]

categorical_features = ['UBICACION_PAG_MIX']


# =========================================================
# ANALISIS DE CORRELACION Y MUTUAL INFORMATION (seleccion de variables)
# Objetivo: detectar variables redundantes antes de la busqueda combinatoria
# =========================================================
from sklearn.feature_selection import mutual_info_regression

# ---- 1. Correlacion de Spearman ----
cols_corr = numeric_features + target
corr_matrix = TRAIN[cols_corr].corr(method='spearman')

print('\n=== 1. CORRELACION DE SPEARMAN (numericas vs target) ===')
print(corr_matrix)

# ---- 2. Mutual Information ----
encoder_mi = CatBoostEncoder(cols=categorical_features, handle_missing="value", handle_unknown="value")
X_mi = TRAIN[numeric_features + categorical_features].copy()
X_mi[categorical_features] = encoder_mi.fit_transform(X_mi[categorical_features], TRAIN[target[0]])
y_mi = TRAIN[target[0]]

mi_scores = mutual_info_regression(X_mi, y_mi, discrete_features=False, random_state=42)

mi_df = pd.DataFrame({
    'variable': X_mi.columns,
    'mutual_info': mi_scores
}).sort_values('mutual_info', ascending=False)

print('\n=== 2. MUTUAL INFORMATION (ordenado de mayor a menor) ===')
print(mi_df.to_string(index=False))

# ---- 3. Comparativo correlacion (numericas) vs mutual information (todas) ----
corr_target = corr_matrix[target[0]].drop(target[0]).abs().rename('corr_abs_spearman')
comparativo = mi_df.set_index('variable').join(corr_target, how='left').sort_values('mutual_info', ascending=False)

print('\n=== 3. COMPARATIVO: MUTUAL INFORMATION vs CORRELACION SPEARMAN ===')
print(comparativo)

# ---- 4. Seleccionar top-k features por Mutual Information ----
if USAR_TOP_K_FEATURES:
    features_seleccionadas = mi_df.head(TOP_K_FEATURES)['variable'].tolist()
    print(f'\n=== 4. TOP {TOP_K_FEATURES} FEATURES SELECCIONADAS (USAR_TOP_K_FEATURES=True) ===')
else:
    features_seleccionadas = numeric_features + categorical_features
    print('\n=== 4. USANDO TODAS LAS FEATURES (USAR_TOP_K_FEATURES=False) ===')
print(features_seleccionadas)


def svs_sbr(q_est, q_real):
    svs = np.where(q_est > q_real, q_real, q_est)
    sbr = np.where(q_est > q_real, q_est - q_real, 0)
    svs_total = round((sum(svs) / sum(q_real)) * 100, 2)
    sbr_total = round((sum(sbr) / sum(q_real)) * 100, 2)
    return svs, sbr, svs_total, sbr_total


def all_combinations(features):
    return list(chain.from_iterable(combinations(features, r) for r in range(1, len(features) + 1)))


def run_model_on_combination(feature_combo):
    num = [f for f in feature_combo if f in numeric_features]
    cat = [f for f in feature_combo if f in categorical_features]
    cyc = [f for f in feature_combo if f in cycle_features]

    try:
        X_train = TRAIN[num + cat + cyc].copy()
        X_test = TEST[num + cat + cyc].copy()
        y_train = TRAIN[target[0]]
        y_test = TEST[target[0]]

        if cyc:
            X_train[cyc] = X_train[cyc].astype(int)
            X_test[cyc] = X_test[cyc].astype(int)

        pre_steps = []

        if cat:
            pre_steps.append(('cat_encoder', CatBoostEncoder(cols=cat, handle_missing="value", handle_unknown="value")))

        col_transformers = []
        if numeric_features:
            col_transformers.append(
                ('num', Pipeline(steps=[('scaler', StandardScaler())]), num))

        if col_transformers:
            pre_steps.append(('col_transform', ColumnTransformer(col_transformers, remainder='passthrough')))

        preprocessor = Pipeline(pre_steps)

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=N_ESTIMATORS_BUSQUEDA, max_depth=19, min_samples_leaf=4, min_samples_split=4, max_features=0.32148581998124864))
        ])

        pipeline.fit(X_train, y_train)

        X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
        estimators = pipeline.named_steps['regressor'].estimators_
        all_predictions = np.array([est.predict(X_test_transformed) for est in estimators])

        percentiles = [2.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 97.5]
        percentile_names = ['LIN_INF', 'Q_10', 'Q_20', 'Q_30', 'Q_40', 'Q_50',
                             'Q_60', 'Q_70', 'Q_80', 'Q_90', 'LIM_SUP']

        predicted_percentiles = np.percentile(all_predictions, percentiles, axis=0)

        X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
        estimators = pipeline.named_steps['regressor'].estimators_
        all_predictions = np.array([est.predict(X_train_transformed) for est in estimators])

        predicted_percentiles_train = np.percentile(all_predictions, percentiles, axis=0)

        num_variables = X_train.shape[1]

        results = []
        for i, name in enumerate(percentile_names):

            # Evaluacion matematica segun el target usado
            if TARGET_SELECCIONADO == 'LOG_RATIO_FACTOR':
                pred_adjusted = np.exp(predicted_percentiles[i]) * TEST['FACTOR_RCP_ORIGEN']
                pred_adjusted_train = np.exp(predicted_percentiles_train[i]) * TRAIN['FACTOR_RCP_ORIGEN']
            elif TARGET_SELECCIONADO == 'RATIO_FACTOR':
                pred_adjusted = predicted_percentiles[i] * TEST['FACTOR_RCP_ORIGEN']
                pred_adjusted_train = predicted_percentiles_train[i] * TRAIN['FACTOR_RCP_ORIGEN']
            else:
                pred_adjusted = predicted_percentiles[i]
                pred_adjusted_train = predicted_percentiles_train[i]

            y_real = 'FACTOR_RCP_DESTINO'

            svs_ind, sbr_ind, srv_total, sbr_total = svs_sbr(pred_adjusted, TEST[y_real])
            svs_ind_train, sbr_ind_train, srv_total_train, sbr_total_train = svs_sbr(pred_adjusted_train, TRAIN[y_real])

            TEST_TEMP = TEST.copy()
            TEST_TEMP['SVS_IND'] = svs_ind
            TEST_TEMP['SBR_IND'] = sbr_ind

            # agrupado por CAMP (campana destino) -- ADV_NAC no tiene CAMP_DESTINO
            grouped = TEST_TEMP.groupby('CAMP').agg({
                'SVS_IND': lambda x: (sum(x) / sum(TEST_TEMP.loc[x.index, y_real])) * 100,
                'SBR_IND': lambda x: (sum(x) / sum(TEST_TEMP.loc[x.index, y_real])) * 100
            })

            std_svs = grouped['SVS_IND'].std()
            std_sbr = grouped['SBR_IND'].std()

            if TARGET_SELECCIONADO == 'LOG_RATIO_FACTOR':
                lim_sup = np.exp(predicted_percentiles[-1]) * TEST['FACTOR_RCP_ORIGEN'].values
                lim_inf = np.exp(predicted_percentiles[0]) * TEST['FACTOR_RCP_ORIGEN'].values
                per_50 = np.exp(predicted_percentiles[5]) * TEST['FACTOR_RCP_ORIGEN'].values
            elif TARGET_SELECCIONADO == 'RATIO_FACTOR':
                lim_sup = predicted_percentiles[-1] * TEST['FACTOR_RCP_ORIGEN'].values
                lim_inf = predicted_percentiles[0] * TEST['FACTOR_RCP_ORIGEN'].values
                per_50 = predicted_percentiles[5] * TEST['FACTOR_RCP_ORIGEN'].values
            else:
                lim_sup = predicted_percentiles[-1]
                lim_inf = predicted_percentiles[0]
                per_50 = predicted_percentiles[5]

            ratio_per_50_lim_inf = sum(per_50) / sum(lim_inf)
            ratio_lim_sup_per_50 = sum(lim_sup) / sum(per_50)

            real_values = TEST[y_real].values
            above_lim_sup = np.sum(real_values > lim_sup)
            below_lim_inf = np.sum(real_values < lim_inf)
            outside_percent = ((above_lim_sup + below_lim_inf) / len(real_values)) * 100

            negativos = np.sum(pred_adjusted < 0)
            total_elementos = len(pred_adjusted)
            porcentaje_negativos = (negativos / total_elementos) * 100

            mae_test = mean_absolute_error(TEST[y_real], pred_adjusted)
            mape_test = np.mean(np.abs((TEST[y_real].values - pred_adjusted) / TEST[y_real].values)) * 100

            mae_train = mean_absolute_error(TRAIN[y_real], pred_adjusted_train)
            mape_train = np.mean(np.abs((TRAIN[y_real].values - pred_adjusted_train) / TRAIN[y_real].values)) * 100

            results.append({
                'MODELO': 'RANDOM FOREST',
                'TARGET': 'RATIO',
                'PERCENTIL': name,
                'NÚMERO DE VARIABLES': num_variables,
                'VARIABLES': feature_combo,
                'SRV_TEST': srv_total,
                'SBR_TEST': sbr_total,
                'SRV_TRAIN': srv_total_train,
                'SBR_TRAIN': sbr_total_train,
                'STD_SRV': round(std_svs, 2),
                'STD_SBR': round(std_sbr, 2),
                'MAE_TEST': round(mae_test, 4),
                'MAPE_TEST': round(mape_test, 2),
                'MAE_TRAIN': round(mae_train, 4),
                'MAPE_TRAIN': round(mape_train, 2),
                'NEGATIVE': negativos,
                'T0TAL_ELEMENTOS': total_elementos,
                'PORCENTAJE_NEGATIVOS': porcentaje_negativos,
                'RATIO_LIMITES_50_INF': ratio_per_50_lim_inf,
                'RATIO_LIMITES_SUP_50': ratio_lim_sup_per_50,
                'DENTRO DE LÍMITES': (100 - round(outside_percent, 2))
            })

        return results

    except Exception as e:
        return [{
            'MODELO': 'RANDOM FOREST',
            'TARGET': 'RATIO',
            'PERCENTIL': name,
            'NÚMERO DE VARIABLES': None,
            'VARIABLES': feature_combo,
            'SRV_TEST': None,
            'SBR_TEST': None,
            'SRV_TRAIN': None,
            'SBR_TRAIN': None,
            'STD_SRV': None,
            'STD_SBR': None,
            'MAE_TEST': None,
            'MAPE_TEST': None,
            'MAE_TRAIN': None,
            'MAPE_TRAIN': None,
            'NEGATIVE': None,
            'T0TAL_ELEMENTOS': None,
            'PORCENTAJE_NEGATIVOS': None,
            'RATIO_LIMITES_50_INF': None,
            'RATIO_LIMITES_SUP_50': None,
            'DENTRO DE LÍMITES': None,
            'Error': str(e)
        } for name in ['LIN_INF', 'Q_10', 'Q_20', 'Q_30', 'Q_40', 'Q_50', 'Q_60', 'Q_70', 'Q_80', 'Q_90', 'LIM_SUP']]


start_time = time.time()
feature_combos = all_combinations(features_seleccionadas + cycle_features)
n_combos = len(feature_combos)

n_cores = os.cpu_count() or 1
t0 = time.time()
primer_resultado = run_model_on_combination(feature_combos[0])
tiempo_por_combo = time.time() - t0
tiempo_estimado_min = (n_combos * tiempo_por_combo) / n_cores / 60

print('\n=== TIEMPO ESTIMADO ===')
print(f'Combinaciones a probar: {n_combos}')
print(f'Tiempo de 1 combinacion (referencia): {tiempo_por_combo:.1f} seg')
print(f'Tiempo estimado total (~{n_cores} cores en paralelo): {tiempo_estimado_min:.1f} min')

resultados_restantes = Parallel(n_jobs=-1, verbose=10, prefer='threads')(
    delayed(run_model_on_combination)(combo) for combo in feature_combos[1:])
results = [primer_resultado] + resultados_restantes

flattened_results = list(chain.from_iterable(results))
results_df = pd.DataFrame(flattened_results)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo de ejecución: {elapsed_time:.4f} segundos")

results_df_export = results_df.copy()
results_df_export['VARIABLES'] = results_df['VARIABLES'].astype(str)
results_df_export["MODELO"] = TARGET_SELECCIONADO
results_df_export.TARGET = TARGET_SELECCIONADO

EXPORTAMOS = results_df_export
count = EXPORTAMOS["MODELO"].value_counts()
print(count)

columnas_clave = ['MODELO', 'TARGET', 'VARIABLES']
EXPORTAMOS['ANDROIDE'] = EXPORTAMOS.groupby(columnas_clave).ngroup()
EXPORTAMOS['ANDROIDE'] = EXPORTAMOS['ANDROIDE'].astype(str)

EXPORTAMOS.to_csv(EXPERIMENTOS_NUBE / f"{comentario}.txt", index=False, sep=',')

# %%
