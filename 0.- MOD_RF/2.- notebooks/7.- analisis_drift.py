"""
Análisis de Data Drift multivariado (MMD) y univariado (PSI) entre Train y Test.
"""
# %%
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Agregar rutas relativas
MOD_RF_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(MOD_RF_DIR / '3.- src'))
ROOT_PROYECTO = MOD_RF_DIR.parents[0]
sys.path.append(str(ROOT_PROYECTO))

from models.data_drift import calcular_mmd, reporte_psi, graficar_reporte_psi, graficar_drift_multivariado_pca
from features.construir_features import obtener_features_candidatas
from config_proyecto import CORTE_TRAIN_TEST

# %% Cargar Datos
ruta_datos = MOD_RF_DIR / '0.- data' / 'processed' / 'df_limpio.parquet'
if not ruta_datos.exists():
    raise FileNotFoundError("Ejecuta entrenar.py primero para generar df_limpio.parquet")

df = pd.read_parquet(ruta_datos)

print(f"Dividiendo datos en Train (< {CORTE_TRAIN_TEST}) y Test (>= {CORTE_TRAIN_TEST})")
df_train = df[df['CAMP'] < int(CORTE_TRAIN_TEST)].copy()
df_test = df[df['CAMP'] >= int(CORTE_TRAIN_TEST)].copy()
print(f"Muestras Train: {len(df_train):,} | Muestras Test: {len(df_test):,}")

# %% Definir variables a analizar
# Obtenemos las variables numericas oficiales del modelo (sin necesidad de variables base)
candidatas = obtener_features_candidatas([]) 
variables_num = candidatas['num']
# Quedarnos solo con las que existen en la tabla
variables_validas = [v for v in variables_num if v in df_train.columns]

#REMOVER PRECIOS ABSOLUTOS Y CANTIDADES PURAS DE LA LISTA
variables_validas = [v for v in variables_validas if "PRECIO_PROMEDIO" not in v and "NUM_PRODUCTOS" not in v]


# %% 1. Analisis Univariado (PSI)
df_psi = reporte_psi(df_train, df_test, variables_validas)
print("\n--- Reporte PSI (Population Stability Index) ---")
print(df_psi.head(10).to_string(index=False))

# Graficar y guardar reporte visual
ruta_grafico_psi = MOD_RF_DIR / '5.- reports' / 'graficos' / 'reporte_drift_psi.png'
ruta_grafico_psi.parent.mkdir(parents=True, exist_ok=True)

graficar_reporte_psi(df_psi, export_path=ruta_grafico_psi)
plt.show()

# %% 2. Analisis Multivariado (MMD)
from sklearn.impute import SimpleImputer
print("\n--- Calculando MMD Multivariado ---")
# Imputamos nulos con la mediana de Train y luego estandarizamos
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_num = imputer.fit_transform(df_train[variables_validas])
X_train_scaled = scaler.fit_transform(X_train_num)

X_test_num = imputer.transform(df_test[variables_validas])
X_test_scaled = scaler.transform(X_test_num)

res_mmd = calcular_mmd(X_train_scaled, X_test_scaled, n_perm=100)
mmd_score = res_mmd['mmd_score']
p_value = res_mmd['p_value']

print(f"Puntaje MMD (Maximum Mean Discrepancy): {mmd_score:.4f}")
print(f"Umbral critico (95%): {res_mmd['umbral_critico_95']:.4f} | p-value: {p_value:.4f}")
print(f"Detalle -> Intra-Train: {res_mmd['k_xx_intra_train']:.4f} | Intra-Test: {res_mmd['k_yy_intra_test']:.4f} | Cruzado: {res_mmd['k_xy_cruzado']:.4f}")

if p_value < 0.05:
    print("⚠️ ALERTA ESTADISTICA: Existe una discrepancia multivariada significativa entre Train y Test (p < 0.05).")
else:
    print("✅ OK: La distribucion multivariada se mantiene estadisticamente estable.")

# %% 2.1 Evolucion de MMD por Campana en Test
print("\n--- Evolucion de Drift por Campana en Test ---")
campanas_test = sorted(df_test['CAMP'].unique())
for camp in campanas_test:
    mask = df_test['CAMP'] == camp
    if mask.sum() > 0:
        X_test_camp = X_test_scaled[mask]
        res_camp = calcular_mmd(X_train_scaled, X_test_camp, gamma=res_mmd['gamma_usado'], n_perm=50)
        p_val_camp = res_camp['p_value']
        estado = "⚠️ ALERTA" if p_val_camp < 0.05 else "✅ OK"
        print(f"Campana {camp} (N={mask.sum():,}): MMD = {res_camp['mmd_score']:.4f} | p-value = {p_val_camp:.4f} | {estado}")


# Graficar y guardar reporte visual PCA
ruta_grafico_pca = MOD_RF_DIR / '5.- reports' / 'graficos' / 'reporte_drift_mmd_pca.png'
graficar_drift_multivariado_pca(X_train_scaled, X_test_scaled, export_path=ruta_grafico_pca)
plt.show()
