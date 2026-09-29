#%%
"""
Rutas y variables de plantilla centralizadas para ROP_ADV_NAC. Todo script
bajo src/, notebooks/ debe importar sus rutas de aqui en vez de hardcodear
absolutas (venian hardcodeadas y repetidas en cada seccion de "0.- MOD_0.py").

Este modelo es distinto a HOG_NAC_NAC/ROP_NAC_NAC: no pronostica un solo
pais, sino que usa la demanda ADVANCE de un pais "origen" (adelantado
DESFASE_CAMPANAS campanias) para explicar la demanda NACIONAL de un pais
"destino". Para adaptar esto a otro par pais/linea, edita PAIS_DESTINO,
PAIS_ORIGEN, LINEA y DESFASE_CAMPANAS abajo.
"""
import sys
from pathlib import Path
import os
import warnings

# Este archivo vive en 0.- MOD_RF/3.- src/config.py:
#   parents[0]=3.- src, parents[1]=0.- MOD_RF, parents[2]=raiz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Par de paises, linea, desfase, ventana de campañas, corte y comentario de la
# corrida: se definen UNA vez en config_proyecto.py (raiz), compartidos con 0.- MOD_TLL
sys.path.append(str(PROJECT_ROOT))
from config_proyecto import (PAIS_DESTINO, PAIS_ORIGEN, LINEA, DESFASE_CAMPANAS,
                             CAMP_INI, CAMP_FIN, CORTE_TRAIN_TEST, COMENTARIO_REG)

# En Windows 11 24H2+ (build 26100 en adelante) ya no existe wmic.exe, que
# joblib usa para contar nucleos FISICOS -- cae a nucleos logicos y emite
# este warning inofensivo (ver joblib/externals/loky/backend/context.py). Se
# ignora aca (import temprano en todos los scripts) para no ensuciar la
# salida de RandomForest/Boruta (n_jobs=-1 en entrenamiento_utils.py).
warnings.filterwarnings('ignore', message='Could not find the number of physical cores')

# --- VARIABLES DE PLANTILLA (editar al adaptar a un nuevo proyecto) --------
NOMBRE_PROYECTO = "AZZORTI"
NOMBRE_MODELO = "ROP_ADV_NAC"
NOMBRE_PKL = f"PIPELINE_{NOMBRE_MODELO}.pkl"  # lo guarda entrenar.py y lo leen los notebooks

# Factores regionales crudos del ADV segun PAIS_ORIGEN. Unico lugar donde se
# definen: lo usan entrenar.py (PASO 5 y features) y los notebooks, para que
# cambiar PAIS_ORIGEN no rompa listas de features copiadas a mano.
FACTORES_REGIONALES_POR_PAIS = {
    'COL': ['FACT_BOG', 'FACT_OCC1', 'FACT_SAN1', 'FACT_COS'],
    'PER': ['FACT_LIMA', 'FACT_COSTA', 'FACT_SIERRA', 'FACT_SELVA'],
}

# --- RUTAS INTERNAS (dentro de la plantilla) -------------------------------
DEV_ROOT = PROJECT_ROOT / '0.- MOD_RF'

DATA_RAW = DEV_ROOT / '0.- data' / 'raw'
DATA_INTERIM = DEV_ROOT / '0.- data' / 'interim'
DATA_PROCESSED = DEV_ROOT / '0.- data' / 'processed'
PRUEBAS_MODELO_DIR = DATA_PROCESSED / 'PRUEBAS DE MODELO'
# copia de cada corrida de referencia en la carpeta de pruebas de 0.- MOD_TLL,
# donde se comparan referencia vs talla (consolidador + Power BI). Solo se
# escribe si el proyecto tiene 0.- MOD_TLL (ver entrenar.py)
TLL_ROOT = PROJECT_ROOT / '0.- MOD_TLL'
PRUEBAS_MODELO_TLL_DIR = TLL_ROOT / '0.- data' / 'processed' / 'PRUEBAS DE MODELO'
MODELS_DIR = DEV_ROOT / '4.- models'
REPORTS_EXCEL = DEV_ROOT / '5.- reports' / 'excel'
REPORTS_FIGURES = DEV_ROOT / '5.- reports' / 'figures'

# --- RUTAS EXTERNAS (fuera de la plantilla, fuentes de datos compartidas) --
# NO son relativas al proyecto: viven en otras carpetas de OneDrive/red
# compartidas por otro equipo/sistema. Unico lugar donde editarlas si cambia
# esa estructura externa.
USER = os.getlogin()
ONEDRIVE_STATS_DTS = Path(f'C:/Users/{USER}/35159_147728_DUPREE_VENTA_DIRECTA_S_A/STATS DTS - Documentos')
ONEDRIVE_STATS = Path(f'C:/Users/{USER}/35159_147728_DUPREE_VENTA_DIRECTA_S_A/STATS - Documentos')

# Mismo diccionario de carpetas por pais que usa MODELO_AZZORTI_HOG_NAC_NAC /
# MODELO_AZZORTI_ROP_NAC_NAC (ver COMO_USAR_PLANTILLA.md de esos proyectos).
DIC_PAISES = {
    'PER': '2.- PER',
    'COL': '3.- COL',
    'ECU': '4.- ECU',
    'BOL': '5.- BOL',
    'GUA': '6.- GUA',
}

RUTA_AZZ_BASE = ONEDRIVE_STATS_DTS / '1.- AZZ'   # VEH_EST (mismo root que ROP_NAC_NAC)
RUTA_DTS_BASE = ONEDRIVE_STATS / '1.- DTS'       # FCT_TZO_AGG / FCT_ADV_GP1 / COD_INF, por pais

# TODO: esta ruta de CON_PRD es la que trae el codigo original (0.- MOD_0.py),
# y es DISTINTA a RUTA_CON_PRD en MODELO_AZZORTI_ROP_NAC_NAC/config.py, que
# apunta a "STATS DTS - Documentos/0.- CMN/1.- CON_PRD/CON_PRD_MOD.snap.parquet"
# (otra carpeta raiz y otra profundidad). Confirmar con el equipo de datos
# cual esta vigente antes de asumir que son intercambiables.
RUTA_CON_PRD = RUTA_DTS_BASE / '1.- CMN' / '2.- GEN' / '1.- CON_PRD' / 'CON_PRD_MOD.snap.parquet'

# La tabla de referencia de competidores (CATEGORIAS por SUBGRUPO) ya no se
# lee de un excel en tiempo de ejecucion -- ver CATEGORIAS_POR_SUBGRUPO en
# 3.- src/data/cargar_datos.py y 1.- references/README.md.

# TODO: "0.- MOD_0.py" escribia con esta misma variable ("path_inpro") en dos
# carpetas distintas segun el punto del script (una con y otra sin el prefijo
# "0.- "). Se deja aqui UNA sola carpeta de intermedios (DATA_INTERIM, dentro
# de la plantilla) para no seguir arrastrando esa inconsistencia; si algun
# consumidor externo todavia lee de las rutas originales de OneDrive, avisar
# antes de cambiar donde se guarda "datos_estimar_5.csv"/"data_5.csv".
