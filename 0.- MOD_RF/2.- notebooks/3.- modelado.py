"""
EDA previo al modelado: distribucion de RATIO_FACTOR (base del target,
LOG_RATIO_FACTOR), deteccion visual de posibles atipicos via
RATIO_EXPOSICION, y comparacion Train/Test de las variables candidatas antes
de correr Optuna. La busqueda de hiperparametros en si (RandomForest +
CatBoostEncoder, mismo enfoque que MODELO_AZZORTI_ROP_NAC_NAC) vive en
3.- src/models/entrenar.py (PASO 9-10) -- este notebook es solo para revisar
visualmente el target y las variables antes de esa busqueda.
"""
#%% Librerias
import sys
from pathlib import Path  # manipular rutas de forma OOP

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

sys.path.append(str(Path(__file__).resolve().parents[1] / "3.- src"))
from config import DATA_PROCESSED
from models.entrenamiento_utils import dividir_train_test_por_corte
from visualization.graficos import graficar_distribucion_ratio, graficar_ratio_vs_exposicion, graficar_variables_vs_target, graficar_importancia_rf

dat_ref = pd.read_csv(DATA_PROCESSED / 'datos_estimar_5.csv', encoding='cp1252')

#%% Distribucion de RATIO_FACTOR con umbral de atipicos (media + 3*std)
umbral = dat_ref['RATIO_FACTOR'].mean() + 3 * dat_ref['RATIO_FACTOR'].std()
graficar_distribucion_ratio(dat_ref, 'RATIO_FACTOR', limite_superior=umbral)

#%% RATIO_FACTOR vs RATIO_EXPOSICION (deteccion visual de posibles atipicos)
graficar_ratio_vs_exposicion(dat_ref, col_ratio='RATIO_FACTOR', col_exposicion='RATIO_EXPOSICION', limite_ratio=umbral)

#%% mismo grafico, coloreado por RATIO_PRECIO
graficar_ratio_vs_exposicion(dat_ref, col_ratio='RATIO_FACTOR', col_exposicion='RATIO_EXPOSICION',
                              col_color='RATIO_PRECIO', limite_ratio=umbral)

#%% Train/Test por corte de campana (ver PASO 8 de entrenar.py)
train, test = dividir_train_test_por_corte(dat_ref, col_camp='CAMP', corte=202508)
df_long = pd.concat([
    train.dropna(subset=['RATIO_FACTOR']).assign(split='Train'),
    test.dropna(subset=['RATIO_FACTOR']).assign(split='Test'),
], ignore_index=True)

vars_continuas = ['FACT_ADV_RGN', 'PRECIO_N', 'PRECIO_ADV_IN', 'RATIO_PRECIO',
                   'PRECIO_PROMEDIO_COMPETIDORES_ORIGEN', 'PRECIO_PROMEDIO_COMPETIDORES_DESTINO',
                   'FACTOR_MAX', 'FACTOR_MIN', 'FACTOR_MEDIANA', 'EXPOSICION_ORIGEN', 'EXPOSICION_DESTINO',
                   'RATIO_EXPOSICION']
vars_discretas = ['GRUPO', 'SUBGRUPO', 'CATEGORIAS', 'CAMBIO_PAG_LADO', 'TIPO_REF']

#%%
graficar_variables_vs_target(df_long, vars_continuas, y_var='RATIO_FACTOR', tipo='continuas')

#%%
graficar_variables_vs_target(df_long, vars_discretas, y_var='RATIO_FACTOR', tipo='discretas')

#%% Importancia rapida de las variables de pagina (sanity check, no es el modelo final)
train_pag = train.copy()
test_pag = test.copy()
le = LabelEncoder()
train_pag['CAMBIO_PAG_LADO_ENC'] = le.fit_transform(train_pag['CAMBIO_PAG_LADO'])

vars_pag = ['PAG_ADV_IN_PAG_DER', 'PAG_NAC_NEW_PAG_DER', 'CAMBIO_PAG_LADO_ENC', 'PAG_ADV_IN', 'PAG_NAC_NEW']
sub = train_pag[vars_pag + ['RATIO_FACTOR']].dropna()

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(sub[vars_pag], sub['RATIO_FACTOR'])
graficar_importancia_rf(rf, vars_pag, titulo='Importancia RF -- Variables de Pagina')
