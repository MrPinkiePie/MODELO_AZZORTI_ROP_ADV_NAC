"""
Explica predicciones puntuales del pipeline ya entrenado y decidido
(4.- models/*.pkl), via SHAP. No reentrena ni compara modelos — eso
se hace en 3.- modelado.py. Mismo notebook que MODELO_AZZORTI_ROP_NAC_NAC;
solo cambia la columna de campaña (CAMP en vez de CAMP_DESTINO).
"""
#%% Librerias
import sys
from pathlib import Path  # manipular rutas de forma OOP

import pandas as pd
import shap
import matplotlib.pyplot as plt
from joblib import load

sys.path.append(str(Path(__file__).resolve().parents[1] / "3.- src"))
from config import NOMBRE_PKL, MODELS_DIR, PRUEBAS_MODELO_DIR

#%% Diagnostico: por que una prediccion puntual es tan variable (SHAP)
# usa el pipeline ya entrenado y el historico de pruebas ya exportados por
# entrenar.py / consolidar_pruebas_modelo.py, sin reentrenar nada
nombre_modelo = NOMBRE_PKL
pipe = load(MODELS_DIR / nombre_modelo)

df_pruebas = pd.read_parquet(PRUEBAS_MODELO_DIR / 'consolidado_pruebas_modelo.parquet')

#%% Elegir la fila puntual a explicar
# CAMP es la campaña destino (NAC) en este modelo -- equivale a CAMP_DESTINO de NAC_NAC
camp_destino = 202508
nombre_sap = 'JEAN BORNEO R4512 TI'

df_pruebas['DISPERSION'] = df_pruebas['LIM_SUP'] - df_pruebas['LIM_INF']
filtro = (df_pruebas['CAMP'] == camp_destino) & (df_pruebas['NOMBRE_SAP'] == nombre_sap)

if filtro.sum() == 0:
    disponibles = sorted(df_pruebas.loc[df_pruebas['NOMBRE_SAP'] == nombre_sap, 'CAMP'].unique())
    raise ValueError(f'No hay fila para CAMP={camp_destino}. '
                      f'Campañas disponibles para "{nombre_sap}": {disponibles}')

idx_fila = df_pruebas.loc[filtro].index[0]
fila = df_pruebas.loc[idx_fila]
print(f'====== FILA A EXPLICAR: {fila["NOMBRE_SAP"]} CAMP={fila["CAMP"]} '
      f'(y_real={fila["y_real"]:.4f}, y_pred={fila["y_pred"]:.4f}, DISPERSION={fila["DISPERSION"]:.4f}) ======')

columnas_modelo = list(pipe.feature_names_in_)  # mismas columnas crudas usadas al hacer fit
X_fila = df_pruebas.loc[[idx_fila], columnas_modelo]

#%% SHAP sobre esa fila puntual
X_fila_enc = pipe.named_steps['prep'].transform(X_fila)

explainer = shap.TreeExplainer(pipe.named_steps['modelo'])
shap_values = explainer(X_fila_enc)
shap_values.feature_names = columnas_modelo

shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
plt.show()
