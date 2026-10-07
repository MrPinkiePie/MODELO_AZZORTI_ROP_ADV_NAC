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

# --- 1.1 Preparación de Variables Descriptivas (Género y Año) ---
if 'CATEGORIAS' in df.columns:
    cat_str = df['CATEGORIAS'].astype(str).str.upper()
    cond_fem = cat_str.str.contains('FEM')
    cond_msc = cat_str.str.contains('MSC')
    df['GENERO_INFERIDO'] = np.select([cond_fem, cond_msc], ['FEMENINO', 'MASCULINO'], default='MIXTO / OTRO')
else:
    df['GENERO_INFERIDO'] = 'N/A'

if 'CAMP' in df.columns:
    df['ANIO'] = df['CAMP'].astype(str).str[:4]

# --- 1.2 Análisis de Outliers (Valores altos de RATIO_FACTOR) ---
print("\n--- 1.2 Exportando Excel de Análisis de Ratio Factor (Outliers y Agrupaciones) ---")
# Parámetros editables para el usuario
UMBRAL_TIPO = 'fijo' # Puede ser 'fijo' o 'percentil'
VALOR_UMBRAL = 8 # Si es fijo, RATIO_FACTOR > 8. Si es percentil, ej. 95 (top 5%)

if UMBRAL_TIPO == 'fijo':
    corte = VALOR_UMBRAL
else:
    corte = np.percentile(df[TARGET_VISUAL].dropna(), VALOR_UMBRAL)

df_outliers = df[df[TARGET_VISUAL] > corte].copy()

# Columnas a exportar
cols_export = ['CAMP', 'CLASIF'] if 'CLASIF' in df.columns else ['CAMP']
cols_export += ['GENERO_INFERIDO', 'SUBGRUPO'] if 'SUBGRUPO' in df.columns else []
cols_export += ['NOMBRE_SAP', TARGET_VISUAL]
cols_export += ['URL_ADV_IN', 'URL_NAC'] if 'URL_NAC' in df.columns else []

# Intersección para evitar errores si no existe alguna
cols_export = [c for c in cols_export if c in df_outliers.columns]

df_outliers = df_outliers[cols_export].sort_values(by=TARGET_VISUAL, ascending=False)

# Crear resumen (Metadatos)
resumen = {
    'Parámetro Usado': f"{UMBRAL_TIPO.upper()} ({VALOR_UMBRAL})",
    'Corte Real Aplicado (Ratio)': round(corte, 2),
    'Rango de Campañas Analizado': f"{df['CAMP'].min()} a {df['CAMP'].max()}" if 'CAMP' in df.columns else "N/A",
    'Total Referencias en Dataset': len(df),
    'Referencias Outliers Encontradas': len(df_outliers),
    'Porcentaje del Dataset': f"{(len(df_outliers) / len(df)):.2%}",
    'Ratio Máximo en Outliers': round(df_outliers[TARGET_VISUAL].max(), 2) if not df_outliers.empty else 0,
    'Ratio Mínimo en Outliers': round(df_outliers[TARGET_VISUAL].min(), 2) if not df_outliers.empty else 0,
    'Promedio Ratio en Outliers': round(df_outliers[TARGET_VISUAL].mean(), 2) if not df_outliers.empty else 0
}
df_resumen = pd.DataFrame(list(resumen.items()), columns=['Métrica', 'Valor'])

# Agregar Ratio Factor Promedio y Mediano por Año al Resumen Técnico
ratio_por_anio = df.groupby('ANIO')[TARGET_VISUAL].agg(['mean', 'median']).round(2)
filas_anio = []
for anio, row in ratio_por_anio.iterrows():
    filas_anio.append({'Métrica': f'Ratio Promedio Año {anio}', 'Valor': row['mean']})
    filas_anio.append({'Métrica': f'Ratio Mediano Año {anio}', 'Valor': row['median']})
df_resumen = pd.concat([df_resumen, pd.DataFrame(filas_anio)], ignore_index=True)


# --- 1.3 Agrupaciones por Campaña y Nivel de Detalle ---
# 1. Agrupacion general (Nivel más alto: la campana completa)
grp_camp = df.groupby(['ANIO', 'CAMP', 'CLASIF'])[TARGET_VISUAL].agg(
    CANT_REFERENCIAS='count',
    RATIO_PROMEDIO='mean',
    RATIO_MEDIANO='median',
    RATIO_MIN='min',
    RATIO_MAX='max'
).reset_index().sort_values(['ANIO', 'CAMP'])

# 2. Agrupacion por Género (Nivel intermedio: Campaña + Género)
grp_gen = df.groupby(['ANIO', 'CAMP', 'CLASIF', 'GENERO_INFERIDO'])[TARGET_VISUAL].agg(
    CANT_REFERENCIAS='count',
    RATIO_PROMEDIO='mean',
    RATIO_MEDIANO='median',
    RATIO_MIN='min',
    RATIO_MAX='max'
).reset_index().sort_values(['ANIO', 'CAMP', 'GENERO_INFERIDO'])

# 3. Agrupacion por Subgrupo (Nivel detallado: Campaña + Subgrupo)
grp_sub = df.groupby(['ANIO', 'CAMP', 'CLASIF', 'SUBGRUPO'])[TARGET_VISUAL].agg(
    CANT_REFERENCIAS='count',
    RATIO_PROMEDIO='mean',
    RATIO_MEDIANO='median',
    RATIO_MIN='min',
    RATIO_MAX='max'
).reset_index().sort_values(['ANIO', 'CAMP', 'SUBGRUPO'])

# 4. Agrupacion por Clima (Campaña + Clima)
if 'CLIMA' in df.columns:
    grp_cli = df.groupby(['ANIO', 'CAMP', 'CLASIF', 'CLIMA'])[TARGET_VISUAL].agg(
        CANT_REFERENCIAS='count',
        RATIO_PROMEDIO='mean',
        RATIO_MEDIANO='median',
        RATIO_MIN='min',
        RATIO_MAX='max'
    ).reset_index().sort_values(['ANIO', 'CAMP', 'CLIMA'])
else:
    grp_cli = pd.DataFrame()

# Exportar a Excel
from config import REPORTS_EXCEL
ruta_outliers = REPORTS_EXCEL / 'Outliers_Ratio_Factor.xlsx'
REPORTS_EXCEL.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(ruta_outliers) as writer:
    df_resumen.to_excel(writer, sheet_name='Resumen_Tecnico', index=False)
    grp_camp.to_excel(writer, sheet_name='Ratio_Por_Campana', index=False)
    grp_gen.to_excel(writer, sheet_name='Ratio_Campana_Genero', index=False)
    grp_sub.to_excel(writer, sheet_name='Ratio_Campana_Subgrupo', index=False)
    if not grp_cli.empty:
        grp_cli.to_excel(writer, sheet_name='Ratio_Campana_Clima', index=False)
    df_outliers.to_excel(writer, sheet_name='Detalle_Outliers', index=False)
print(f"✅ Excel de Análisis de Ratio Factor exportado en: {ruta_outliers}\n")

# Boxplot del target por subgrupo (para ver diferencias entre familias)
if 'SUBGRUPO' in df.columns:
    plt.figure(figsize=(12, 6))
    orden = df.groupby('SUBGRUPO')[TARGET_VISUAL].median().sort_values().index
    sns.boxplot(data=df, x='SUBGRUPO', y=TARGET_VISUAL, order=orden, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title("Distribución del Target (RATIO_FACTOR) según el SUBGRUPO")
    plt.tight_layout()
    plt.show()

    # Análisis por Categoría (Femenino, Masculino, etc.)
    if 'CATEGORIAS' in df.columns:
        plt.figure(figsize=(10, 6))
        orden_cat = df.groupby('CATEGORIAS')[TARGET_VISUAL].median().sort_values().index
        sns.boxplot(data=df, x='CATEGORIAS', y=TARGET_VISUAL, order=orden_cat, palette='Set2')
        plt.title("Distribución del Target por CATEGORIAS")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # 1.1 Inferir Género (Femenino / Masculino / Otro) desde CATEGORIAS
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
        plt.title("Distribución del Target (RATIO_FACTOR) según GÉNERO")
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
    'CAMP_SEN', 'CAMP_COS', 'MOMENTUM_SUBGRUPO_LAG_3',
    'MOMENTUM_CLIMA_LAG_3', 'HHI_CLIMA_ORIGEN'
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

# Añadimos TARGET_MODEL a la lista para el heatmap como pidió el usuario
cols_corr = [c for c in features_fijas_num if c in df.columns]
if TARGET_VISUAL in df.columns:
    cols_corr.append(TARGET_VISUAL)

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
                mask=mask, annot_kws={"size": 10}, linewidths=0.5)
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
