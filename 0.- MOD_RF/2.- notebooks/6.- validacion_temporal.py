#%%
"""
Validador del punto de corte temporal (CORTE_TRAIN_TEST).
Usa la nueva funcion graficar_learning_curve_temporal para evaluar que 
tan estable es el modelo al cambiar la campana que divide Train y Test.
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

sys.path.append(str(Path(__file__).resolve().parents[1] / "3.- src"))
from config import DATA_PROCESSED
from visualization.graficos import graficar_learning_curve_temporal

# 1. Cargar la data final ya procesada
# Asegurate de haber corrido entrenar.py antes para que el archivo exista
dat_ref = pd.read_csv(DATA_PROCESSED / 'PRUEBAS DE MODELO' / 'df_COL_GUA_REX_PRUEBA 1.csv') # Ajusta el nombre si cambia

#%%
# 2. Definir las variables y el Pipeline base
# Reemplaza estas listas con las variables reales que tengas en tu entrenar.py
features_num = ['RATIO_PRECIO', 'FACT_ADV_RGN', 'PRECIO_N'] 
features_cat = ['SUBGRUPO', 'CATEGORIAS']

pipeline = Pipeline([
    ('encoder', ce.CatBoostEncoder(cols=features_cat)),
    ('model', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
])

# 3. Graficar la curva
# Si no le pasas cortes_candidatos, probara automaticamente el 50% central de tus campanas
ax, resultados = graficar_learning_curve_temporal(
    df=dat_ref, 
    pipeline=pipeline, 
    features_num=features_num, 
    features_cat=features_cat, 
    col_camp='CAMP', 
    target_col='LOG_RATIO_FACTOR'
)

plt.show()

#%%
# 4. Ver los resultados exactos en tabla
print(resultados.sort_values('Test_Score', ascending=False))
