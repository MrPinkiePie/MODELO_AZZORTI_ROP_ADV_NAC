"""
Análisis Exploratorio de Datos (EDA) sobre el dataset final consolidado de Features vs Target.
Usa df_limpio.parquet que ya contiene el target (LOG_RATIO_FACTOR) cruzado con todas las X (features numéricas y categóricas).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

# Asegurar importes desde src
sys.path.append(str(Path(__file__).resolve().parents[1] / "3.- src"))
from config import DATA_PROCESSED

# Configuración visual
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Cargar datos
print("Cargando dataset limpio...")
df = pd.read_parquet(DATA_PROCESSED / 'df_limpio.parquet')
print(f"Dataset cargado: {df.shape[0]:,} filas y {df.shape[1]} columnas.")

# El target es LOG_RATIO_FACTOR (ya está en escala logarítmica para entrenar)
# o FACTOR_RCP_DESTINO (escala real). Usaremos la escala log (target del modelo).
TARGET = 'LOG_RATIO_FACTOR'

# -------------------------------------------------------------
# SECCION 1: ANÁLISIS DE LA VARIABLE OBJETIVO
# -------------------------------------------------------------
print("\n--- 1. Analizando Variable Objetivo ---")
plt.figure()
sns.histplot(df[TARGET], kde=True, bins=50, color='blue')
plt.title("Distribución de la Variable Objetivo (LOG_RATIO_FACTOR)")
plt.xlabel("Log(Ratio)")
plt.ylabel("Frecuencia")
plt.show()

# Boxplot del target por subgrupo (para ver diferencias entre familias)
if 'SUBGRUPO' in df.columns:
    plt.figure(figsize=(12, 6))
    orden = df.groupby('SUBGRUPO')[TARGET].median().sort_values().index
    sns.boxplot(data=df, x='SUBGRUPO', y=TARGET, order=orden, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title("Distribución del Target según el SUBGRUPO")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------
# SECCION 2: CORRELACIÓN LINEAL (HEATMAP)
# -------------------------------------------------------------
print("\n--- 2. Mapa de Calor (Correlación Pearson) ---")
# Filtramos solo numéricas
cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

# Calculamos matriz de correlación (quitando columnas de IDs irrelevantes si existen)
excluir = ['CAMP', 'ANIO'] 
cols_corr = [c for c in cols_numericas if c not in excluir]

corr_matrix = df[cols_corr].corr(method='pearson')

# Extraer correlaciones solo con el Target
if TARGET in corr_matrix:
    corr_target = corr_matrix[[TARGET]].sort_values(by=TARGET, ascending=False)
    print("Correlación con el Target:")
    print(corr_target)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False)
plt.title("Mapa de Calor - Correlaciones Lineales de Variables Numéricas")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# SECCION 3: MUTUAL INFORMATION (RELACIONES NO LINEALES)
# -------------------------------------------------------------
print("\n--- 3. Mutual Information (Todas las Features) ---")
# Mutual info requiere que no haya Nulos y maneja las categóricas si se codifican
# Para este EDA rápido usaremos las numéricas que no sean target ni factores directos equivalentes
df_mi = df.dropna(subset=[TARGET] + cols_corr).copy()
X = df_mi[cols_corr].drop(columns=[TARGET, 'FACTOR_RCP_DESTINO', 'FACTOR_RCP_ORIGEN', 'RATIO_FACTOR'], errors='ignore')
y = df_mi[TARGET]

if not X.empty:
    # Calcular Mutual Information
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    mi_series.plot(kind='bar', color='teal')
    plt.title("Mutual Information (Dependencia no lineal) contra Target")
    plt.ylabel("Score MI")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print("Top 5 variables con mayor Mutual Information:")
    print(mi_series.head(5))

    # -------------------------------------------------------------
    # SECCION 4: SCATTER PLOTS (LAS MEJORES VARIABLES)
    # -------------------------------------------------------------
    top_vars = mi_series.head(3).index.tolist()
    print(f"\n--- 4. Scatter Plots de las Mejores Variables ({top_vars}) ---")

    for var in top_vars:
        plt.figure()
        # Usamos sample si son demasiados puntos
        df_sample = df.sample(n=min(5000, len(df)), random_state=42)
        sns.scatterplot(data=df_sample, x=var, y=TARGET, alpha=0.3, color='purple')
        plt.title(f"Relación: {var} vs {TARGET}")
        plt.show()
