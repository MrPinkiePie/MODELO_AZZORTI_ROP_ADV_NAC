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
# SECCION 2: MUTUAL INFORMATION (RELACIONES NO LINEALES)
# -------------------------------------------------------------
print("\n--- 2. Mutual Information (Todas las Features) ---")
# Mutual info requiere que no haya Nulos y maneja las categóricas si se codifican
# Filtramos solo las variables predictoras (X) numéricas reales que usa el modelo 
# (tomado de FEATURES_FIJAS['num'] en entrenar.py)
features_fijas_num = [
    'RATIO_PRECIO', 'FACT_ADV_RGN', 'FACTOR_MEDIANA', 'FACTOR_MAX', 'FACTOR_MIN',
    'PRECIO_PROMEDIO_COMPETIDORES_ORIGEN', 'NUM_PRODUCTOS_POR_CATEGORIAS_ORIGEN',
    'RATIO_PROMEDIO_COMPETIDORES_PRECIO_ORIGEN',
    'PRECIO_PROMEDIO_COMPETIDORES_DESTINO', 'NUM_PRODUCTOS_POR_CATEGORIAS_DESTINO',
    'PAG_NAC_NEW', 'PAG_ADV_IN',
    'CAMP_SEN', 'CAMP_COS', 'MOMENTUM_SUBGRUPO_LAG_3'
]

cols_mi = [c for c in features_fijas_num if c in df.columns]

df_mi = df.dropna(subset=[TARGET_MODEL] + cols_mi).copy()
X = df_mi[cols_mi].drop(columns=['FACTOR_RCP_DESTINO', 'FACTOR_RCP_ORIGEN', 'FAC_NAC', 'FACT_ADV_RGN'], errors='ignore')
y = df_mi[TARGET_MODEL] # MI se calcula contra el target de entrenamiento

if not X.empty:
    from sklearn.feature_selection import mutual_info_regression
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

    # Elegiremos el Top 9 para el análisis de dispersión posterior
    top_vars_9 = mi_series.head(9).index.tolist()
else:
    top_vars_9 = cols_mi[:9]

# %%
# -------------------------------------------------------------
# SECCION 3: DISTRIBUCIÓN Y DISPERSIÓN DE LAS X's (OUTLIERS)
# -------------------------------------------------------------
print("\n--- 3. Análisis Univariado de las X's (Histogramas y Boxplots) ---")

vars_to_plot = top_vars_9 

# Histogramas para ver la distribución (asimetría, campana de Gauss)
plt.figure(figsize=(15, 10))
for i, col in enumerate(vars_to_plot, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True, color='skyblue', bins=30)
    plt.title(f"Histograma: {col}", fontsize=10)
    plt.xlabel("")
plt.suptitle("Distribución del Top 9 Variables (por Mutual Info)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Boxplots para ver dispersión pura y valores atípicos (Outliers)
plt.figure(figsize=(15, 10))
for i, col in enumerate(vars_to_plot, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col], color='lightgreen')
    plt.title(f"Boxplot: {col}", fontsize=10)
    plt.ylabel("")
plt.suptitle("Dispersión del Top 9 Variables (por Mutual Info)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %%
# -------------------------------------------------------------
# SECCION 4: CORRELACIÓN LINEAL (HEATMAP)
# -------------------------------------------------------------
print("\n--- 4. Mapa de Calor (Correlación Pearson) ---")

# Usamos exactamente las mismas variables de FEATURES_FIJAS para el mapa de calor
cols_corr = [c for c in features_fijas_num if c in df.columns]

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
    plt.grid(False)  # Quitar el grid para no ensuciar la visualización
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
# SECCION 5: SCATTER PLOTS (LAS MEJORES VARIABLES)
# -------------------------------------------------------------
# Usaremos el top 3 del Mutual Information calculado en la Seccion 2
if 'mi_series' in locals():
    top_vars = mi_series.head(3).index.tolist()
    print(f"\n--- 5. Scatter Plots de las Mejores Variables ({top_vars}) ---")

    for var in top_vars:
        plt.figure()
        df_sample = df.sample(n=min(3000, len(df)), random_state=42)
        sns.regplot(data=df_sample, x=var, y=TARGET_VISUAL, 
                    scatter_kws={'alpha': 0.3, 'color': 'purple'},
                    line_kws={'color': 'red'}, lowess=True)
        plt.title(f"Relación y Tendencia: {var} vs {TARGET_VISUAL}")
        plt.show()

# %%
# -------------------------------------------------------------
# SECCION 6: VARIABLES CLAVE DE NEGOCIO (PRECIO, ESTRATEGIA, HISTORIA)
# -------------------------------------------------------------
print("\n--- 6. Impacto de Variables de Negocio en la Demanda ---")

# 6.1 Impacto del Precio Relativo (RATIO_PRECIO)
if 'RATIO_PRECIO' in df.columns:
    plt.figure(figsize=(10, 6))
    df_sample = df.sample(n=min(3000, len(df)), random_state=42)
    sns.regplot(data=df_sample, x='RATIO_PRECIO', y=TARGET_VISUAL, 
                scatter_kws={'alpha': 0.3, 'color': 'navy'},
                line_kws={'color': 'darkorange'}, lowess=True)
    plt.title(f"Impacto del Precio Relativo en el Factor\n(RATIO_PRECIO vs {TARGET_VISUAL})")
    plt.xlabel("RATIO_PRECIO (Precio Destino / Precio Origen)")
    plt.ylabel(TARGET_VISUAL)
    plt.show()

# 6.2 Estrategia Mixta (ESTR_MIX)
if 'ESTR_MIX' in df.columns:
    plt.figure(figsize=(12, 6))
    orden = df.groupby('ESTR_MIX')[TARGET_VISUAL].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='ESTR_MIX', y=TARGET_VISUAL, order=orden, palette='Set3')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.title(f"Distribución del Factor por Estrategia de Venta (ESTR_MIX)")
    plt.tight_layout()
    plt.show()

# 6.3 Ubicación en Catálogo (UBICACION_PAG_MIX)
if 'UBICACION_PAG_MIX' in df.columns:
    plt.figure(figsize=(10, 5))
    orden = df.groupby('UBICACION_PAG_MIX')[TARGET_VISUAL].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='UBICACION_PAG_MIX', y=TARGET_VISUAL, order=orden, palette='Pastel1')
    plt.title(f"Impacto de la Ubicación en el Catálogo")
    plt.tight_layout()
    plt.show()

# 6.4 Tipo de Referencia (NUEVA vs CON_HISTORIA)
if 'TIPO_REF' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='TIPO_REF', y=TARGET_VISUAL, palette='Set1')
    plt.title(f"Comparación del Factor: Productos Nuevos vs Con Historia")
    plt.tight_layout()
    plt.show()
