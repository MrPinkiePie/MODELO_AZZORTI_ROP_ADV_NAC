# %%
"""
Análisis Exploratorio de Datos (EDA) sobre Estacionalidad del Target.
Revisa el comportamiento de RATIO_FACTOR a lo largo del tiempo, por campaña, por año,
y por festividades (Día de la Madre, Día del Padre).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Asegurar importes desde src
sys.path.append(str(Path(__file__).resolve().parents[1] / "3.- src"))
from config import DATA_PROCESSED

# Configuración visual
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("Cargando dataset limpio...")
df = pd.read_parquet(DATA_PROCESSED / 'df_limpio.parquet')
print(f"Dataset cargado: {df.shape[0]:,} filas y {df.shape[1]} columnas.")

# TARGET_VISUAL: usamos la escala real para visualizar la estacionalidad claramente
TARGET = 'RATIO_FACTOR'

# Asegurarnos de que NUM_CAMP sea numérico para que el eje X se ordene correctamente
if 'NUM_CAMP' in df.columns:
    df['NUM_CAMP'] = df['NUM_CAMP'].astype(int)
else:
    df['NUM_CAMP'] = df['CAMP'].astype(str).str[-2:].astype(int)

if 'ANIO' not in df.columns:
    df['ANIO'] = df['CAMP'].astype(str).str[:4].astype(int)

# Crear variable de Género Inferido si no existe
if 'GENERO_INFERIDO' not in df.columns and 'CATEGORIAS' in df.columns:
    cat_str = df['CATEGORIAS'].astype(str).str.upper()
    df['GENERO_INFERIDO'] = np.select(
        [cat_str.str.contains('FEM'), cat_str.str.contains('MSC')], 
        [f'FEMENINO', f'MASCULINO'], 
        default='MIXTO / OTRO'
    )

# %%
# -------------------------------------------------------------
# SECCION 1: ESTACIONALIDAD POR AÑO Y CAMPAÑA (LINEPLOT)
# -------------------------------------------------------------
print("\n--- 1. Estacionalidad Promedio por Campaña y Año ---")
# Agrupamos por año y número de campaña para sacar el promedio del ratio
df_season = df.groupby(['ANIO', 'NUM_CAMP'])[TARGET].mean().reset_index()
df_season = df_season.sort_values(by=['ANIO', 'NUM_CAMP'])

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_season, x='NUM_CAMP', y=TARGET, hue='ANIO', marker='o', palette='tab10')
plt.title(f"Comportamiento de {TARGET} a lo largo del año (por Campaña de Destino)")
plt.xlabel("Número de Campaña (1 a 18)")
plt.ylabel(f"Promedio de {TARGET}")
plt.xticks(range(1, 19))
plt.legend(title='Año')
plt.tight_layout()
plt.show()

# %%
# -------------------------------------------------------------
# SECCION 2: DESCOMPOSICIÓN ESTACIONAL (SEASONAL DECOMPOSE)
# -------------------------------------------------------------
print("\n--- 2. Descomposición Estacional (Tendencia, Estacionalidad, Residuos) ---")
# Creamos una serie de tiempo continua (CAMP en orden cronológico)
df_ts = df.groupby('CAMP')[TARGET].mean().sort_index()

# Verificamos que tengamos al menos 2 periodos completos (36 campañas) para una buena descomposición
if len(df_ts) >= 36:
    # Period=18 porque hay 18 campañas en un año
    result = seasonal_decompose(df_ts.values, model='additive', period=18)
    
    # Creamos un eje X continuo (0, 1, 2...) para evitar los "huecos" numéricos entre 202418 y 202501
    x_axis = range(len(df_ts))
    
    # Grid 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Original (Top-Left)
    axes[0, 0].plot(x_axis, df_ts.values, label='Original')
    axes[0, 0].set_title("1. Serie de Tiempo Original")
    axes[0, 0].set_xticks(x_axis[::4])
    axes[0, 0].set_xticklabels(df_ts.index.astype(str)[::4], rotation=45)
    
    # 2. Tendencia (Top-Right)
    axes[0, 1].plot(x_axis, result.trend, label='Tendencia', color='orange')
    axes[0, 1].set_title("2. Tendencia (Trend) - Media Móvil")
    axes[0, 1].set_xticks(x_axis[::4])
    axes[0, 1].set_xticklabels(df_ts.index.astype(str)[::4], rotation=45)
    
    # 3. Estacionalidad (Bottom-Left)
    axes[1, 0].plot(x_axis, result.seasonal, label='Estacionalidad', color='green')
    axes[1, 0].set_title("3. Estacionalidad (Pura)")
    axes[1, 0].set_xticks(x_axis[::4])
    axes[1, 0].set_xticklabels(df_ts.index.astype(str)[::4], rotation=45)
    
    # 4. Residuos (Bottom-Right)
    axes[1, 1].plot(x_axis, result.resid, label='Residuos', color='red', marker='o', linestyle='none')
    axes[1, 1].set_title("4. Residuos (Anomalías)")
    axes[1, 1].set_xticks(x_axis[::4])
    axes[1, 1].set_xticklabels(df_ts.index.astype(str)[::4], rotation=45)
    
    plt.tight_layout()
    plt.show()
else:
    print(f"No hay suficientes campañas ({len(df_ts)}) para hacer seasonal_decompose con period=18. Se necesitan al menos 36.")

# %%
# -------------------------------------------------------------
# SECCION 3: EVENTOS ESPECIALES (DIA DE LA MADRE Y PADRE) POR CATEGORIAS
# -------------------------------------------------------------
print("\n--- 3. Efecto de Festividades por Categoría ---")
if 'CATEGORIAS' in df.columns:
    df_cat = df[df['CATEGORIAS'].isin(['FEMENINO', 'MASCULINO'])]
    if df_cat.empty:
        # En caso que se llamen de otra forma (FEM, MSC)
        cat_counts = df['CATEGORIAS'].value_counts()
        print(f"Las categorías disponibles son: {cat_counts.index.tolist()}")
        df_cat = df.copy() # Usamos todas si no hay explícitamente MASCULINO/FEMENINO
        
    df_cat_season = df_cat.groupby(['CATEGORIAS', 'NUM_CAMP'])[TARGET].mean().reset_index()

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df_cat_season, x='NUM_CAMP', y=TARGET, hue='CATEGORIAS', marker='o', palette='Set2')
    
    # Sombreado para Día de la Madre (Campañas 4 a 7)
    plt.axvspan(4, 7, color='magenta', alpha=0.1, label='Temporada Día de la Madre')
    
    # Sombreado para Día del Padre (Campañas 8, 9, 10)
    plt.axvspan(8, 10, color='blue', alpha=0.1, label='Temporada Día del Padre')
    
    # Sombreado para Fin de Año (Campañas 16, 17, 18)
    plt.axvspan(16, 18, color='grey', alpha=0.1, label='Temporada Fin de Año')

    plt.title(f"Estacionalidad de {TARGET} por Categoría y Festividades")
    plt.xlabel("Número de Campaña (1 a 18)")
    plt.ylabel(f"Promedio de {TARGET}")
    plt.xticks(range(1, 19))
    plt.legend(title='Categoría / Evento')
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------
# SECCION 3B: EVENTOS ESPECIALES (DIA DE LA MADRE Y PADRE) POR GÉNERO
# -------------------------------------------------------------
print("\n--- 3B. Efecto de Festividades por Género (Agrupado) ---")
if 'GENERO_INFERIDO' in df.columns:
    df_gender = df[df['GENERO_INFERIDO'].isin(['FEMENINO', 'MASCULINO'])]
    df_gender_season = df_gender.groupby(['GENERO_INFERIDO', 'NUM_CAMP'])[TARGET].mean().reset_index()

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df_gender_season, x='NUM_CAMP', y=TARGET, hue='GENERO_INFERIDO', marker='o', palette={'FEMENINO': 'magenta', 'MASCULINO': 'blue'})
    
    # Sombreado para Día de la Madre (Campañas 4 a 7)
    plt.axvspan(4, 7, color='magenta', alpha=0.1, label='Temporada Día de la Madre')
    
    # Sombreado para Día del Padre (Campañas 8, 9, 10)
    plt.axvspan(8, 10, color='blue', alpha=0.1, label='Temporada Día del Padre')
    
    # Sombreado para Fin de Año (Campañas 16, 17, 18)
    plt.axvspan(16, 18, color='grey', alpha=0.1, label='Temporada Fin de Año')

    plt.title(f"Estacionalidad de {TARGET} por Género y Festividades")
    plt.xlabel("Número de Campaña (1 a 18)")
    plt.ylabel(f"Promedio de {TARGET}")
    plt.xticks(range(1, 19))
    plt.legend(title='Género / Evento')
    plt.tight_layout()
    plt.show()

# %%
# -------------------------------------------------------------
# SECCION 4: ESTACIONALIDAD CLIMÁTICA (CON ARCHIVO EXTERNO)
# -------------------------------------------------------------
print("\n--- 4. Efecto de Estacionalidad Climática ---")
if 'CLIMA' in df.columns:
    if 'SUBGRUPO' in df.columns:
        subg_str = df['SUBGRUPO'].astype(str)
        df.loc[subg_str.str.startswith('209'), 'CLIMA'] = 'FRIO'
        df.loc[subg_str.str.startswith('210'), 'CLIMA'] = 'NEUTRO'
        df.loc[subg_str.str.startswith('230') | subg_str.str.startswith('227'), 'CLIMA'] = 'CALIDO'
        
    df_clima_season = df.groupby(['CLIMA', 'NUM_CAMP'])[TARGET].mean().reset_index()
    
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df_clima_season, x='NUM_CAMP', y=TARGET, hue='CLIMA', marker='s', linewidth=2)
    plt.title(f"Estacionalidad de {TARGET} por Tipo de Clima / Temporada")
    plt.xlabel("Número de Campaña (1 a 18)")
    plt.ylabel(f"Promedio de {TARGET}")
    plt.xticks(range(1, 19))
    plt.legend(title='Clima')
    plt.tight_layout()
    plt.show()
else:
    print("La columna CLIMA no está en el dataset. Asegúrate de correr entrenar.py primero.")

