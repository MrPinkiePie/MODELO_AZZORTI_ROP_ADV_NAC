"""
Analisis visual de RATIO_PRECIO (precio destino/origen): distribucion,
relacion con RATIO_FACTOR (base del target), y evolucion en el tiempo. Migrado desde
"analisis_ratio_precio.py" (usaba una ruta de OneDrive distinta a donde
"0.- MOD_0.py" guardaba el mismo archivo -- ver TODO en config.py sobre esa
inconsistencia; aca se lee del CSV que exporta entrenar.py en 0.- data/processed/).
"""
#%% Librerias
import sys
from pathlib import Path  # manipular rutas de forma OOP

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "3.- src"))
from config import DATA_PROCESSED
from visualization.graficos import graficar_evolucion_ratio_precio

dat_ref = pd.read_csv(DATA_PROCESSED / 'datos_estimar_5.csv', encoding='cp1252')

#%% 1. Distribucion general
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(dat_ref['RATIO_PRECIO'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(dat_ref['RATIO_PRECIO'].mean(), color='red', linestyle='--', linewidth=2,
           label=f"Media: {dat_ref['RATIO_PRECIO'].mean():.3f}")
ax.axvline(dat_ref['RATIO_PRECIO'].median(), color='green', linestyle='--', linewidth=2,
           label=f"Mediana: {dat_ref['RATIO_PRECIO'].median():.3f}")
ax.set_xlabel('RATIO_PRECIO (destino/origen)')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribucion de RATIO_PRECIO')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(dat_ref['RATIO_PRECIO'].describe())

#%% 2. RATIO_PRECIO vs RATIO_FACTOR (target)
import seaborn as sns

fig, ax = plt.subplots(figsize=(12, 7))
sns.regplot(data=dat_ref, x='RATIO_PRECIO', y='RATIO_FACTOR',
            scatter_kws={'alpha': 0.6, 's': 100, 'edgecolor': 'black', 'linewidths': 0.5},
            line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2}, ax=ax)
ax.set_title('Relacion RATIO_PRECIO vs RATIO_FACTOR')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print('Correlacion:', dat_ref['RATIO_PRECIO'].corr(dat_ref['RATIO_FACTOR']))

#%% 3. Evolucion de RATIO_PRECIO por campana
graficar_evolucion_ratio_precio(dat_ref, col_camp='CAMP', col_ratio='RATIO_PRECIO', step=5)

#%% 4. Evolucion de RATIO_FACTOR por campana
graficar_evolucion_ratio_precio(dat_ref, col_camp='CAMP', col_ratio='RATIO_FACTOR', step=5)

