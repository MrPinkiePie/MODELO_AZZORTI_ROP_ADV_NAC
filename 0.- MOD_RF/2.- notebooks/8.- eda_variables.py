"""
Análisis Exploratorio de Datos (EDA) sobre el dataset final consolidado de Features vs Target.
Usa df_limpio.parquet que ya contiene el target (LOG_RATIO_FACTOR) cruzado con todas las X (features numéricas y categóricas).
"""
# %%
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

# El target modelo es LOG_RATIO_FACTOR, pero para interpretación visual usaremos RATIO_FACTOR
TARGET_MODEL = 'LOG_RATIO_FACTOR'
TARGET_VISUAL = 'RATIO_FACTOR'

# %%
# -------------------------------------------------------------
# SECCION 1: ANÁLISIS DE LA VARIABLE OBJETIVO
# -------------------------------------------------------------
print("\n--- 1. Analizando Variable Objetivo ---")
plt.figure()
sns.histplot(df[TARGET_VISUAL], kde=True, bins=50, color='blue')
plt.title("Distribución de la Variable Objetivo (RATIO_FACTOR)")
plt.xlabel("Ratio")
plt.ylabel("Frecuencia")
plt.show()

# Boxplot del target por subgrupo (para ver diferencias entre familias)
if 'SUBGRUPO' in df.columns:
    plt.figure(figsize=(12, 6))
    orden = df.groupby('SUBGRUPO')[TARGET_VISUAL].median().sort_values().index
    sns.boxplot(data=df, x='SUBGRUPO', y=TARGET_VISUAL, order=orden, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title("Distribución del Target (RATIO_FACTOR) según el SUBGRUPO")
    plt.tight_layout()
    plt.show()

    # Inferir Género (Femenino / Masculino / Otro) desde CATEGORIAS
    if 'CATEGORIAS' in df.columns:
        # Se asume que la columna tiene texto. Pasamos a string por si acaso.
        cat_str = df['CATEGORIAS'].astype(str).str.upper()
        cond_fem = cat_str.str.contains('FEM')
        cond_msc = cat_str.str.contains('MSC')
        
        df['GENERO_INFERIDO'] = np.select(
            [cond_fem, cond_msc], 
            ['FEMENINO', 'MASCULINO'], 
            default='MIXTO / OTRO'
        )
        
        plt.figure(figsize=(8, 5))
        orden_gen = df.groupby('GENERO_INFERIDO')[TARGET_VISUAL].median().sort_values().index
        sns.boxplot(data=df, x='GENERO_INFERIDO', y=TARGET_VISUAL, order=orden_gen, palette='Set2')
        plt.title("Distribución del Target (RATIO_FACTOR) según GÉNERO INFERIDO")
        plt.tight_layout()
        plt.show()

# %%
# -------------------------------------------------------------
# SECCION 2: CORRELACIÓN LINEAL (HEATMAP)
# -------------------------------------------------------------
print("\n--- 2. Mapa de Calor (Correlación Pearson) ---")
# Filtramos solo numéricas
cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

# Calculamos matriz de correlación (quitando columnas de IDs irrelevantes si existen)
excluir = ['CAMP', 'ANIO'] 
cols_corr = [c for c in cols_numericas if c not in excluir]

def plot_heatmap(df_data, columnas, target_col, method='pearson'):
    """Genera un mapa de calor dinámico sin barra lateral y con anotaciones."""
    corr_matrix = df_data[columnas].corr(method=method)
    
    if target_col in corr_matrix:
        corr_target = corr_matrix[[target_col]].sort_values(by=target_col, ascending=False)
        print(f"Correlación ({method.capitalize()}) con el Target:")
        print(corr_target)

    plt.figure(figsize=(14, 12))
    
    # Máscara para ocultar la mitad superior (triángulo superior) y evitar info duplicada
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=True, fmt=".2f", cbar=False, 
                mask=mask, annot_kws={"size": 6}, linewidths=0.5)
    plt.title(f"Mapa de Calor - Correlaciones ({method.capitalize()}) de Variables Numéricas")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

plot_heatmap(df, cols_corr, TARGET_VISUAL, method='pearson')
# Si quieres ver Spearman, simplemente descomenta la siguiente línea:
# plot_heatmap(df, cols_corr, TARGET_VISUAL, method='spearman')

# %%
# -------------------------------------------------------------
# SECCION 3: MUTUAL INFORMATION (RELACIONES NO LINEALES)
# -------------------------------------------------------------
print("\n--- 3. Mutual Information (Todas las Features) ---")
# Mutual info requiere que no haya Nulos y maneja las categóricas si se codifican
# Para este EDA rápido usaremos las numéricas que no sean target ni factores directos equivalentes
df_mi = df.dropna(subset=[TARGET_MODEL] + cols_corr).copy()
X = df_mi[cols_corr].drop(columns=[TARGET_MODEL, TARGET_VISUAL, 'FACTOR_RCP_DESTINO', 'FACTOR_RCP_ORIGEN'], errors='ignore')
y = df_mi[TARGET_MODEL] # MI se calcula contra el target de entrenamiento

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
        df_sample = df.sample(n=min(3000, len(df)), random_state=42)
        
        # Usamos regplot en lugar de scatterplot para tener una linea de tendencia.
        # lowess=True ajusta una curva flexible a los datos (ideal si la relación no es una recta perfecta)
        # scatter_kws aplica transparencia a los puntos para ver densidad.
        sns.regplot(data=df_sample, x=var, y=TARGET_VISUAL, 
                    scatter_kws={'alpha': 0.3, 'color': 'purple'},
                    line_kws={'color': 'red'}, lowess=True)
                    
        plt.title(f"Relación y Tendencia: {var} vs {TARGET_VISUAL}")
        plt.show()
