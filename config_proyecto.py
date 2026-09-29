#%%
"""
Parametros COMUNES a todos los modelos del proyecto: referencia
(0.- MOD_RF) y tallas (0.- MOD_TLL). Unico lugar donde se editan.

Deben coincidir entre ambos modelos porque el de tallas reparte el pronostico
del de referencia: si cambia el par de paises, el rango de campañas o el
corte, las referencias de un lado no cruzan con las del otro (y la
validacion deja de ser comparable).

Cada 3.- src/config.py lo importa y re-exporta, asi que los scripts siguen
haciendo `from config import PAIS_DESTINO, ...` como siempre.
"""

# --- PAR DE PAISES / LINEA -------------------------------------------------
PAIS_DESTINO = 'GUA'   # NAC: demanda nacional que se quiere pronosticar
PAIS_ORIGEN = 'COL'    # ADV: demanda advance/anticipada que explica al NAC
LINEA = 'REX'
DESFASE_CAMPANAS = 3   # el ADV de una campana en PAIS_ORIGEN anticipa el NAC de esa campana+3 en PAIS_DESTINO

# --- VENTANA DE CAMPAÑAS Y SPLIT -------------------------------------------
CAMP_INI = 202401      # Rango inicial actualizado (Sliding window) para evitar Drift de 2022-2023
CAMP_FIN = 202613
CORTE_TRAIN_TEST = 202512  # >= esta campaña queda como TEST (validacion)

# --- CORRIDA DE REFERENCIA -------------------------------------------------
# comentario de la corrida de 0.- MOD_RF: va en el nombre del csv que exporta
# (PRUEBAS DE MODELO) y es el que lee 0.- MOD_TLL para repartir entre tallas
COMENTARIO_REG = 'PRUEBA RAPIDA'
