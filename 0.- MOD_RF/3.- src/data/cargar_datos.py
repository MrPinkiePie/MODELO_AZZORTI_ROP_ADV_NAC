"""
Funciones para cargar las tablas de ROP_ADV_NAC: 1.1 (FCT_TZO_AGG) de destino
y origen, 1.2 (FCT_ADV_GP1) de origen, VEHICULO/ESTRATEGIA, inflados (COD_INF)
y CON_PRD. La tabla de referencia de competidores (CATEGORIAS por SUBGRUPO)
ya no se lee de un excel -- ver CATEGORIAS_POR_SUBGRUPO/mapear_categorias_por_subgrupo.

Migradas desde "0.- MOD_0.py" (secciones de carga de ECU "NAC" y COL "ADV").
A diferencia de MODELO_AZZORTI_ROP_NAC_NAC, este modelo cruza DOS paises
(ver PAIS_DESTINO/PAIS_ORIGEN en config.py), asi que cada funcion recibe el
pais explicito en vez de una lista de paises.

load_1_1 y load_vehiculo_estrategia usan el mismo helper _load_generic (y
la misma fuente RUTA_AZZ_BASE) que MODELO_AZZORTI_HOG_NAC_NAC/ROP_NAC_NAC --
"0.- MOD_0.py" original leia la 1.1 desde RUTA_DTS_BASE en formato .txt, pero
es la MISMA tabla exportada tambien como .snap.parquet en RUTA_AZZ_BASE
(mismo shape, mismas columnas, verificado archivo por archivo): se unifico a
esa fuente por ser la que usan los otros dos modelos y por venir en parquet
(mas rapido que leer .txt/CSV).
"""
#%% Librerias
import glob
import sys
from pathlib import Path  # manipular rutas de forma OOP

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RUTA_AZZ_BASE, RUTA_DTS_BASE, RUTA_CON_PRD, DIC_PAISES
from data.mis_funciones import limpiar_codigo


# Array de campanas validas (anio 2010-2049, campanas 1-18). Mismo patron que
# MODELO_AZZORTI_HOG_NAC_NAC/ROP_NAC_NAC.
CAMPANAS_VALIDAS = np.array([anio * 100 + camp
                              for anio in range(2010, 2050)
                              for camp in range(1, 19)])


def _validar_y_filtrar_campanas(camp_inicio, camp_end):
    """Valida el rango de campanas y devuelve las campanas validas dentro de el. Copiada de MODELO_AZZORTI_HOG_NAC_NAC."""
    try:
        camp_inicio = int(camp_inicio)
        camp_end = int(camp_end)
    except (ValueError, TypeError) as e:
        raise TypeError(f'Las campanas deben ser convertibles a enteros: {e}')

    if camp_inicio > camp_end:
        raise ValueError(f'camp_inicio ({camp_inicio}) debe ser <= camp_end ({camp_end})')

    camps = CAMPANAS_VALIDAS[(CAMPANAS_VALIDAS >= camp_inicio) & (CAMPANAS_VALIDAS <= camp_end)]
    if len(camps) == 0:
        raise ValueError(f'No hay campanas validas entre {camp_inicio} y {camp_end}')
    return camps


def _load_generic(pais, camp_inicio, camp_end, relative_path, filename_pattern, loader_func=pd.read_parquet, verbose=True):
    """
    Carga y concatena un archivo por campana para UN pais (a diferencia de la
    version de MODELO_AZZORTI_HOG_NAC_NAC/ROP_NAC_NAC, que recibe una lista de
    paises -- aca no hace falta porque cada funcion publica ya recibe el pais
    explicito, ver PAIS_DESTINO/PAIS_ORIGEN en config.py).
    """
    camps = _validar_y_filtrar_campanas(camp_inicio, camp_end)
    base_path = RUTA_AZZ_BASE / DIC_PAISES[pais] / relative_path

    df_all = []
    archivos_fallidos = []
    for camp in camps:
        file_path = base_path / filename_pattern.format(pais=pais, camp=camp)
        try:
            df_all.append(loader_func(file_path))
        except FileNotFoundError:
            archivos_fallidos.append(camp)
        except Exception as e:
            if verbose:
                print(f'❌ Error al cargar {pais} - Campana {camp}: {type(e).__name__}: {e}')
            archivos_fallidos.append(camp)

    if len(df_all) == 0:
        raise FileNotFoundError(f'No se encontraron archivos para {pais} en {base_path} '
                                 f'entre {camp_inicio} y {camp_end}')

    df_result = pd.concat(df_all, ignore_index=True)
    if verbose:
        print(f'✅ {pais} {relative_path.split("/")[-1]}: {len(df_all)} archivos '
              f'({len(archivos_fallidos)} no encontrados) | {len(df_result)} filas')
    return df_result


# Columnas de la tabla 1.1 usadas en este modelo. ECU (destino) no trae los
# campos "_A" (son de facturacion advance, no aplican a un pais que solo se
# usa como NAC); COL (origen) si los trae y se usan para el lado ADV.
COLUMNAS_1_1 = [
    'PAIS', 'CAMP', 'CODI_VENT', 'Q_FALT_A', 'Q_FALT_N',
    'Q_PEDI_A', 'Q_PEDI_N', 'LINEA_FACT', 'LINEA_FIN',
    'PRECIO_A', 'PRECIO_N', 'PAG_ADV_NEW', 'PAG_NAC_NEW',
    'TALLA_NEW', 'TALLA_STD', 'GRUPO', 'SUBGRUPO',
    'NOMBRE_SAP', 'NOMBRE_SID', 'SUBUEN', 'TOT_A', 'TOT_N',
    'CORTES_CON_FALT', 'Q_MKT_A', 'Q_MKT_N', 'URL_ADV',
    'URL_NAC', 'CAMP_CODI_VENT', 'VENTA_FCT_A', 'VENTA_FCT_N',
    'VENTA_RCP_A', 'VENTA_RCP_N', 'FAMILIA_TALLA',
    'TOT_PED_MKT_ADV', 'TOT_PED_MKT_NAC',
]


def load_1_1(pais, camp_inicio, camp_end, linea=None, verbose=True):
    """
    Carga y concatena un .snap.parquet por campana de la tabla 1.1
    (RUTA_AZZ_BASE / DIC_PAISES[pais] / '2.- GEN' / '1.1.- FCT_TZO_AGG'), via
    _load_generic -- mismo patron y fuente que load_1_1 de
    MODELO_AZZORTI_HOG_NAC_NAC/ROP_NAC_NAC. Se queda solo con las columnas de
    COLUMNAS_1_1 que existan en el archivo (ECU no trae los campos "_A").

    linea=None NO filtra por LINEA_FIN (deja todas las lineas). En "0.- MOD_0.py"
    el filtro de linea se aplicaba en momentos distintos para destino y origen
    (destino: DESPUES de construir features de competencia por pagina, asi
    que esas features cuentan competidores de TODAS las lineas en esa pagina;
    origen: ANTES, solo cuenta dentro de la misma linea) -- por eso el filtro
    se deja como paso explicito en entrenar.py en vez de aplicarse aca siempre,
    para no perder esa diferencia de orden.
    """
    df = _load_generic(
        pais=pais, camp_inicio=camp_inicio, camp_end=camp_end,
        relative_path='2.- GEN/1.1.- FCT_TZO_AGG',
        filename_pattern='{pais}_FCT_TZO_AGG_{camp}.snap.parquet',
        loader_func=pd.read_parquet, verbose=False)

    columnas = [c for c in COLUMNAS_1_1 if c in df.columns]
    df = df[columnas].copy()

    for col in ['CAMP', 'CODI_VENT', 'PAIS', 'NOMBRE_SAP', 'SUBGRUPO']:
        df[col] = df[col].astype(str)
    df['CODI_VENT'] = limpiar_codigo(df['CODI_VENT'])

    if linea is not None:
        df = df[df['LINEA_FIN'].isin([linea])].copy()
    if verbose:
        print(f'✅ {pais} 1.1: {len(df)} filas | LINEA_FIN={linea}: filtro aplicado')
    return df


def load_vehiculo_estrategia(pais, camp_inicio, camp_end, verbose=True):
    """
    Carga y concatena un .csv por campana de VEHICULO/ESTRATEGIA
    (RUTA_AZZ_BASE / DIC_PAISES[pais] / '4.- ADH' / '7.- VEH_EST'), via
    _load_generic -- mismo patron y fuente que load_vehiculo_estrategia de
    MODELO_AZZORTI_HOG_NAC_NAC/ROP_NAC_NAC (antes esta funcion se llamaba
    load_vehiculo_estrategia_historico y leia TODO el .csv de la carpeta con
    os.listdir en vez de pedir un rango de campanas explicito).
    """
    df = _load_generic(
        pais=pais, camp_inicio=camp_inicio, camp_end=camp_end,
        relative_path='4.- ADH/7.- VEH_EST',
        filename_pattern='{pais}_LEA_LST_{camp}.csv',
        loader_func=pd.read_csv, verbose=verbose)

    df = df.drop_duplicates(['CAMP', 'CODI_VENT'], keep='first')
    df[['PAIS', 'CAMP', 'CODI_VENT']] = df[['PAIS', 'CAMP', 'CODI_VENT']].astype(str)
    df = df.drop(columns=['PAIS'])
    df['CAMP_CODI_VENT'] = df['CAMP'] + '_' + df['CODI_VENT']
    return df


def load_inflados(pais, tipo_zona, verbose=True):
    """
    Carga los .txt de inflados/atipicos (COD_INF) de un pais y filtra por
    TIPO_ZONA ('N' = nacional/destino, 'A' = advance/origen) — a diferencia
    de MODELO_AZZORTI_ROP_NAC_NAC (un solo pais, sin distincion de zona), este
    modelo cruza destino y origen y necesita cada tabla de inflados por
    separado. Devuelve el df con CAMP_NOMBRE_SAP y CAMP_CODI_VENT ya armados
    (requiere CON_PRD para el NOMBRE_SAP).
    """
    path = RUTA_DTS_BASE / DIC_PAISES[pais] / '4.- ADH' / '14.- COD_INF'
    archivos = glob.glob(str(path / '*.txt'))
    if len(archivos) == 0:
        raise FileNotFoundError(f'No se encontraron archivos de inflados para {pais} en {path}')

    df = pd.concat((pd.read_csv(f, sep=',') for f in archivos), ignore_index=True)
    df = df[df['TIPO_ZONA'] == tipo_zona].copy()
    df['CODI_VENT'] = limpiar_codigo(df['CODI_VENT'])

    con_prod = load_cons_prod()
    df = pd.merge(df, con_prod, how='inner', on='CODI_VENT')
    df['CAMP_NOMBRE_SAP'] = df['CAMP'].astype(str) + '_' + df['NOMBRE_SAP']
    df['CAMP_CODI_VENT'] = df['CAMP'].astype(str) + '_' + df['CODI_VENT']

    if verbose:
        print(f'✅ {pais} inflados (TIPO_ZONA={tipo_zona}): {len(archivos)} archivos | {len(df)} filas')
    return df


def load_cons_prod(columnas=('CODI_VENT', 'NOMBRE_SAP')):
    """
    Carga CON_PRD (ver TODO de RUTA_CON_PRD en config.py sobre la
    inconsistencia de ruta contra MODELO_AZZORTI_ROP_NAC_NAC). Mismo nombre
    que load_cons_prod de MODELO_AZZORTI_HOG_NAC_NAC/ROP_NAC_NAC, con
    columnas parametrizable porque ADV_NAC pide distintas columnas segun el
    llamado (ver entrenar.py).
    """
    df = pd.read_parquet(RUTA_CON_PRD)
    df = df[list(columnas)].astype(str)
    return df


# Grupos de factores regionales de advance por pais -- cada tupla agrupa las
# zonas (columnas Q_<zona>/TOT_<zona> de FCT_ADV_GP1) que se suman antes de
# dividir. COL usa codigos de zona logistica (BOG2/BOG3 y COS1/COS3 son
# sub-codigos de una misma zona, por eso se combinan en FACT_BOG/FACT_COS;
# OCC1/SAN1 no tienen sub-codigo asi que quedan solas). Los sub-codigos
# individuales (FACT_BOG2/FACT_BOG3/FACT_COS1/FACT_COS3) no se guardan aparte
# porque nada los usa despues de agruparlos -- solo se necesitan las zonas
# crudas BOG2/BOG3/COS1/COS3 para calcular el combinado. PER usa nombres de
# departamento --sin sub-codigos repetidos-- por eso aca se agrupa por
# macro-region oficial INEI (Costa/Sierra/Selva), dejando LIMA aparte por su
# peso dominante (mismo rol que BOG en COL).
GRUPOS_FACT_ADV_RGN = {
    'COL': {
        'FACT_OCC1': ('OCC1',),  # cali/medellin
        'FACT_SAN1': ('SAN1',),
        'FACT_COS': ('COS1', 'COS3'),
        'FACT_BOG': ('BOG2', 'BOG3'),
    },
    'PER': {
        'FACT_LIMA': ('LIMA',),
        'FACT_COSTA': ('ANCASH', 'LA LIBERTAD'),
        'FACT_SIERRA': ('JUNIN', 'HUANCAVELICA'),
        'FACT_SELVA': ('LORETO',),
    },
}


def leer_fct_adv_gp1(pais):
    """
    Solo la parte de IO de load_fct_adv_gp1(): lee y concatena los parquet de
    la tabla 1.2 (FCT_ADV_GP1) del pais origen (uno por campana, en
    RUTA_DTS_BASE / DIC_PAISES[pais] / '2.- GEN' / '1.2.- FCT_ADV_GP1'). El
    calculo de factores regionales esta en calcular_factores_adv_gp1().
    """
    if pais not in GRUPOS_FACT_ADV_RGN:
        raise KeyError(f'FCT_ADV_GP1 solo esta definido para {list(GRUPOS_FACT_ADV_RGN)}, no para {pais!r}.')

    path = RUTA_DTS_BASE / DIC_PAISES[pais] / '2.- GEN' / '1.2.- FCT_ADV_GP1'
    archivos = glob.glob(str(path / '*.parquet'))
    archivos = [f for f in archivos if not any(s in f for s in ('DUPE', 'DUCO', 'LAPTOP'))]
    if len(archivos) == 0:
        raise FileNotFoundError(f'No se encontraron archivos 1.2 para {pais} en {path}')

    # normaliza nombres de columna: algunos historicos de PER (anteriores a
    # 2021) guardaron "LA.LIBERTAD" con punto en vez de "LA LIBERTAD" (espacio,
    # nombre real del departamento) -- se unifica antes de concatenar para no
    # duplicar la columna con datos repartidos entre las dos variantes.
    partes = [pd.read_parquet(f) for f in archivos]
    for parte in partes:
        parte.columns = parte.columns.str.replace('.', ' ', regex=False)
    df = pd.concat(partes, ignore_index=True)
    return df, archivos


def calcular_factores_adv_gp1(df, pais):
    """
    A partir del crudo de FCT_ADV_GP1 (ver _leer_fct_adv_gp1), calcula los
    factores regionales de advance: Q_ADV_RGN/TOT_ADV_RGN mas los grupos de
    GRUPOS_FACT_ADV_RGN[pais] (distintos por pais porque COL y PER traen
    columnas de zona distintas -- ver comentario de ese diccionario).
    """
    # zonas unicas sin duplicados: BOG2/BOG3/COS1/COS3 aparecen dos veces en
    # COL (como zona individual y dentro del grupo combinado FACT_BOG/FACT_COS)
    zonas_vistas = set()
    for grupo in GRUPOS_FACT_ADV_RGN[pais].values():
        zonas_vistas.update(grupo)
    zonas = sorted(zonas_vistas)
    columnas = ['CAMP', 'CODI_VENT', 'Q_ADV_RGN', 'TOT_ADV_RGN']
    columnas += [f'Q_{zona}' for zona in zonas] + [f'TOT_{zona}' for zona in zonas]
    df = df[columnas].copy()
    df[['CAMP', 'CODI_VENT']] = df[['CAMP', 'CODI_VENT']].astype(str)
    df['CODI_VENT'] = df['CODI_VENT'].str.strip()

    factores = pd.DataFrame({
        'CAMP_CODI_VENT': df['CAMP'] + '_' + df['CODI_VENT'],
        'CAMP': df['CAMP'],
        'CODI_VENT': df['CODI_VENT'],
        'FACT_ADV_RGN': df['Q_ADV_RGN'] / df['TOT_ADV_RGN'],
    })
    for nombre_fact, zonas_grupo in GRUPOS_FACT_ADV_RGN[pais].items():
        q_total = sum(df[f'Q_{zona}'] for zona in zonas_grupo)
        tot_total = sum(df[f'TOT_{zona}'] for zona in zonas_grupo)
        factores[nombre_fact] = q_total / tot_total
    return factores


def revisar_calidad_fct_adv_gp1(df, pais, verbose=True):
    """
    Chequeo de calidad SOLO INFORMATIVO del crudo de FCT_ADV_GP1 (no filtra
    filas ni lanza excepciones, no detiene el pipeline) -- soporta tanto COL
    como PER via GRUPOS_FACT_ADV_RGN[pais]. La suma de las zonas crudas
    (BOG2/BOG3/COS1/COS3/OCC1/SAN1 en COL; LIMA/ANCASH/LA LIBERTAD/JUNIN/
    HUANCAVELICA/LORETO en PER) en teoria deberia calzar exacto con Q_ADV_RGN
    (el total ya agregado); si no calza -- por arriba o por abajo -- hay algo
    raro en la tabla fuente (se encontro ~11% de filas asi en PER en una
    revision exploratoria de 2022). Imprime el % de filas por diagnostico,
    para decision manual del usuario.
    """
    if pais not in GRUPOS_FACT_ADV_RGN:
        return

    zonas_vistas = set()
    for grupo in GRUPOS_FACT_ADV_RGN[pais].values():
        zonas_vistas.update(grupo)
    q_cols = [f'Q_{zona}' for zona in sorted(zonas_vistas)]

    suma_zonas = df[q_cols].sum(axis=1)
    suma_distinta_al_total = ~np.isclose(suma_zonas, df['Q_ADV_RGN'])
    diagnostico = pd.Series(np.select(
        [df['Q_ADV_RGN'] == 0, suma_distinta_al_total],
        ['ADV_RGN_CERO', 'SUMA_DEPTOS_DISTINTA_AL_TOTAL'],
        default='OK',
    ), index=df.index)

    if verbose:
        resumen = diagnostico.value_counts(normalize=True).mul(100).round(2)
        print(f'ℹ️ Calidad {pais} FCT_ADV_GP1 (informativo, no filtra nada) -- % de filas por diagnostico:')
        print(resumen.to_string())


def load_fct_adv_gp1(pais, verbose=True):
    """
    Orquesta leer_fct_adv_gp1() (IO) + calcular_factores_adv_gp1() (calculo)
    para la tabla 1.2 (FCT_ADV_GP1) del pais origen. Solo PER y COL tienen
    esta tabla (unicos paises "origen" validos de ROP_ADV_NAC). Sin
    equivalente en MODELO_AZZORTI_ROP_NAC_NAC (tabla exclusiva de este modelo).
    """
    df, archivos = leer_fct_adv_gp1(pais)
    factores = calcular_factores_adv_gp1(df, pais)

    if verbose:
        print(f'✅ {pais} 1.2 FCT_ADV_GP1: {len(archivos)} archivos | {len(factores)} filas')
    return factores


# CATEGORIAS por SUBGRUPO, por linea -- reemplaza a load_subs_competidores(),
# que antes leia SUBS_COMPETIDORES_RIN.xlsx en cada corrida (RUTA_SUBS_COMPETIDORES
# en config.py). El mapeo no cambia entre corridas, asi que se congela aca
# como diccionario (generado una sola vez desde ese mismo excel, columna
# GENERO_USO no se usaba en ningun lado y se descarta). Se usa para las
# features de competencia tanto de origen como de destino.
CATEGORIAS_POR_SUBGRUPO = {
    'RIN': {
        '127 RI-MEDIAS FEM': 'ACCESORIOS',
        '104 RI-BUZOS MSC': 'DEPORTIVO_MSC',
        '100 RI-BERMUDAS MSC': 'INFERIOR_EXT_MSC',
        '125 RI-VESTIDOS CONTROL FEM': 'FAJAS_CONTROL_FEM',
        '127 RI-MEDIAS CONTROL FEM': 'FAJAS_CONTROL_FEM',
        '136 RI-SOBRETODOS FEM': 'SUPERIOR_EXT_FEM',
        '137 RI-FALDAS': 'INFERIOR_EXT_FEM',
        '105 RI-CHAQUETAS MSC': 'SUPERIOR_EXT_MSC',
        '108 RI-CINTURILLA': 'FAJAS_CONTROL_FEM',
        '130 RI-PESCADOR CTRL': 'FAJAS_CONTROL_FEM',
        '112 RI-BOXER FEM': 'PANTIES_FEM',
        '139 RI-VESTIDO BANO': 'BANO',
        '116 RI - SALIDA BANO': 'BANO',
        '103 RI-CAPRI CONTROL FEM': 'FAJAS_CONTROL_FEM',
        '131 RI-CICLISTA CTRL': 'FAJAS_CONTROL_FEM',
        '142 RI-BERMUDAS FEM': 'INFERIOR_EXT_FEM',
        '103 RI-ACCESORIOS FEM': 'ACCESORIOS',
        '114 RI-BRASIERES': 'SOSTENES_FEM',
        '129 RI-PANTIES': 'PANTIES_FEM',
        '107 RI-CAMISETAS MSC': 'SUPERIOR_EXT_MSC',
        '107 RI-CAMISETAS FEM': 'SUPERIOR_EXT_FEM',
        '115 RI-TOPS FEM': 'SOSTENES_FEM',
        '101 RI-BLUSAS CONTROL FEM': 'FAJAS_CONTROL_FEM',
        '102 RI-BODYS CONTROL FEM': 'FAJAS_CONTROL_FEM',
        '133 RI-ENTERIZO FEM': 'COMPLETO_FEM',
        '105 RI-BABYDOLLS': 'INT_FEM',
        '101 RI-BLUSAS FEM': 'SUPERIOR_EXT_FEM',
        '118 RI-LEGGINS FEM': 'DEPORTIVO_FEM',
        '106 RI-BATAS FEM': 'DESCANSO',
        '120 RI-PANTALONES FEM': 'INFERIOR_EXT_FEM',
        '134 RI-VESTIDOS FEM': 'COMPLETO_FEM',
        '109 RI-CAPRIS FEM': 'INFERIOR_EXT_FEM',
        '130 RI-CHAQUETAS FEM': 'SUPERIOR_EXT_FEM',
        '113 RI-CONJUNTOS FEM': 'COMPLETO_FEM',
        '124 RI-FAJAS CONTROL FEM': 'FAJAS_CONTROL_FEM',
        '128 RI-MULTIUSOS FEM': 'INT_FEM',
        '121 RI-PESCADORES FEM': 'INFERIOR_EXT_FEM',
        '123 RI-SHORTS FEM': 'INFERIOR_EXT_FEM',
        '112 RI-BOXERS MSC': 'INT_MSC',
        '126 RI-LEVANTADORAS FEM': 'DESCANSO',
        '108 RI-CAMISILLAS MSC': 'INT_MSC',
        '129 RI-PANTIES CONTROL FEM': 'FAJAS_CONTROL_FEM',
        '111 RI-BATOLAS FEM': 'DESCANSO',
        '135 RI-BUZOS FEM': 'DEPORTIVO_FEM',
        '102 RI-BODYS FEM': 'INT_FEM',
        '128 RI-MULTIUSOS MSC': 'INT_MSC',
        '119 RI-CINTURILLAS CONTROL FEM': 'FAJAS_CONTROL_FEM',
        '131 RI-JOGGER FEM': 'DEPORTIVO_FEM',
        '117 RI-JOGGERS MSC': 'DEPORTIVO_MSC',
        '104 RI-CAMIBUZOS FEM': 'SUPERIOR_EXT_FEM',
        '122 RI-ESTILIZADORES CONTROL FEM': 'FAJAS_CONTROL_FEM',
        '118 RI-LEGGINS CONTROL FEM': 'FAJAS_CONTROL_FEM',
        '123 RI-SHORTS CONTROL FEM': 'FAJAS_CONTROL_FEM',
        '110 RI-CHALECOS CONTROL FEM': 'FAJAS_CONTROL_FEM',
        '138 RI-PIJAMAS FEM': 'DESCANSO',
        '141 RI-CAMISILLA FEM': 'INT_FEM',
    },
    # REX: generado igual que RIN (una sola vez, desde los SUBGRUPO reales de
    # la tabla 1.1 con LINEA_FIN='REX' en los 5 paises -- adultos "RE-" y
    # tambien infantil "KA-"/NINA, "KO-"/NINO, que comparten LINEA_FIN='REX'
    # en esta fuente aunque no compartan prefijo). Quedan fuera a proposito
    # (no son categorias de prenda, no se listan aca): PAQUETONES,
    # TAPABOCAS, y los insumos/empaque que a veces aparecen con transacciones
    # reales (CUELLOS Y PUNOS, ELASTICOS, INSUMOS SEMIELABORADOS, BOLSAS
    # MATERIALES DE EMPAQUE) -- si aparecen se listan en el aviso "sin match".
    'REX': {
        '001 KA-BLUSAS NINA': 'SUPERIOR_EXT_NINA',
        '001 KO-CAMISETAS NINO': 'SUPERIOR_EXT_NINO',
        '002 KA-SOBRETODOS NINA': 'SUPERIOR_EXT_NINA',
        '002 KO-BUZOS NINO': 'DEPORTIVO_NINO',
        '003 KA-BODYS NINA': 'SUPERIOR_EXT_NINA',
        '003 KO-CAMISAS NINO': 'SUPERIOR_EXT_NINO',
        '004 KA-BUZOS NINA': 'DEPORTIVO_NINA',
        '004 KO-CAMIBUZOS NINO': 'SUPERIOR_EXT_NINO',
        '005 KA-CAMISAS NINA': 'SUPERIOR_EXT_NINA',
        '005 KO-CHAQUETAS NINO': 'SUPERIOR_EXT_NINO',
        '006 KA-CAMISETAS NINA': 'SUPERIOR_EXT_NINA',
        '006 KO-POLOS NINO': 'SUPERIOR_EXT_NINO',
        '007 KA-CHALECOS NINA': 'SUPERIOR_EXT_NINA',
        '007 KO-BERMUDAS NINO': 'INFERIOR_EXT_NINO',
        '008 KA-CHAQUETAS NINA': 'SUPERIOR_EXT_NINA',
        '008 KO-JEANS NINO': 'INFERIOR_EXT_NINO',
        '009 KO-JOGGERS NINO': 'DEPORTIVO_NINO',
        '010 KA-CONJUNTOS NINA': 'COMPLETO_NINA',
        '010 KO-PANTALONES NINO': 'INFERIOR_EXT_NINO',
        '011 KA-ENTERIZOS NINA': 'COMPLETO_NINA',
        '011 KO-CAMISILLAS NINO': 'SUPERIOR_EXT_NINO',
        '012 KA-OVEROLES NINA': 'COMPLETO_NINA',
        '012 KO-CHALECOS NINO': 'SUPERIOR_EXT_NINO',
        '013 KA-VESTIDOS NINA': 'COMPLETO_NINA',
        '013 KO-CONJUNTOS NINO': 'COMPLETO_NINO',
        '014 KO-OVEROLES NINO': 'COMPLETO_NINO',
        '015 KA-FALDAS NINA': 'INFERIOR_EXT_NINA',
        '015 KO-ACCESORIOS NINO': 'ACCESORIOS_NINO',
        '016 KA-JOGGERS NINA': 'DEPORTIVO_NINA',
        '017 KA-LEGGINS NINA': 'DEPORTIVO_NINA',
        '018 KA-PANTALONES NINA': 'INFERIOR_EXT_NINA',
        '019 KA-SHORTS NINA': 'INFERIOR_EXT_NINA',
        '020 KA-JEANS NINA': 'INFERIOR_EXT_NINA',
        '024 KA-PESCADORES NINA': 'INFERIOR_EXT_NINA',
        '026 KA-CAMIBUZOS NINA': 'SUPERIOR_EXT_NINA',
        '028 KA-ACCESORIO NIN': 'ACCESORIOS_NINA',  # genero truncado en la fuente; prefijo KA- = nina
        '200 RE-BERMUDAS MSC': 'INFERIOR_EXT_MSC',
        '201 RE-BLUSAS FEM': 'SUPERIOR_EXT_FEM',
        '201 RE-CONJUNTOS MSC': 'COMPLETO_MSC',
        '202 RE-BODYS FEM': 'SUPERIOR_EXT_FEM',
        '204 RE-BUZOS FEM': 'DEPORTIVO_FEM',
        '204 RE-BUZOS MSC': 'DEPORTIVO_MSC',
        '205 RE-CAMIBUZOS MSC': 'SUPERIOR_EXT_MSC',
        '206 RE-CAMISAS FEM': 'SUPERIOR_EXT_FEM',
        '206 RE-CAMISAS MSC': 'SUPERIOR_EXT_MSC',
        '207 RE-CAMISETAS FEM': 'SUPERIOR_EXT_FEM',
        '207 RE-CAMISETAS MSC': 'SUPERIOR_EXT_MSC',
        '208 RE-CAMISILLAS FEM': 'SUPERIOR_EXT_FEM',
        '208 RE-CAMISILLAS MSC': 'SUPERIOR_EXT_MSC',
        '209 RE-CAPRIS FEM': 'INFERIOR_EXT_FEM',
        '209 RE-SOBRETODO MSC': 'SUPERIOR_EXT_MSC',
        '210 RE-CHALECOS FEM': 'SUPERIOR_EXT_FEM',
        '210 RE-CHALECOS MSC': 'SUPERIOR_EXT_MSC',
        '211 RE-CHAQUETAS FEM': 'SUPERIOR_EXT_FEM',
        '211 RE-CHAQUETAS MSC': 'SUPERIOR_EXT_MSC',
        '212 RE-CINTURONES FEM': 'ACCESORIOS_FEM',
        '213 RE-CONJUNTOS FEM': 'COMPLETO_FEM',
        '214 RE-ENTERIZOS FEM': 'COMPLETO_FEM',
        '215 RE-FALDAS FEM': 'INFERIOR_EXT_FEM',
        '216 RE-JEANS FEM': 'INFERIOR_EXT_FEM',
        '216 RE-JEANS MSC': 'INFERIOR_EXT_MSC',
        '217 RE-JOGGERS FEM': 'DEPORTIVO_FEM',
        '217 RE-JOGGERS MSC': 'DEPORTIVO_MSC',
        '218 RE-LEGGINS FEM': 'DEPORTIVO_FEM',
        '219 RE-OVEROLES FEM': 'COMPLETO_FEM',
        '220 RE-PANTALONES FEM': 'INFERIOR_EXT_FEM',
        '220 RE-PANTALONES MSC': 'INFERIOR_EXT_MSC',
        '221 RE-PESCADORES FEM': 'INFERIOR_EXT_FEM',
        '222 RE-POLOS FEM': 'SUPERIOR_EXT_FEM',
        '222 RE-POLOS MSC': 'SUPERIOR_EXT_MSC',
        '223 RE-SHORTS FEM': 'INFERIOR_EXT_FEM',
        '224 RE-SOBRETODOS FEM': 'SUPERIOR_EXT_FEM',
        '225 RE-PANOLETAS MSC': 'ACCESORIOS_MSC',
        '225 RE-VESTIDOS FEM': 'COMPLETO_FEM',
        '227 RE- TOPS FEM': 'SUPERIOR_EXT_FEM',
        '228 RE-PANOLETAS FEM': 'ACCESORIOS_FEM',
        '229 RE-CAMIBUZOS FEM': 'SUPERIOR_EXT_FEM',
        '230 RE-BERMUDAS FEM': 'INFERIOR_EXT_FEM',
        '231 RE-ACCESORIOS': 'ACCESORIOS',
        '232 RE-BLUSONES FEM': 'SUPERIOR_EXT_FEM',
        '233 RE-CROPTOP FEM': 'SUPERIOR_EXT_FEM',
    },
}


def mapear_categorias_por_subgrupo(df, linea, col_subgrupo='SUBGRUPO', col_linea_filtro='LINEA_FIN', nueva_col='CATEGORIAS'):
    """
    Asigna CATEGORIAS a partir de SUBGRUPO por lookup directo en el
    diccionario congelado CATEGORIAS_POR_SUBGRUPO[linea] (ver arriba) --
    mismo mecanismo (.map) para cualquier linea, RIN y REX incluidas.
    Imprime los SUBGRUPO sin match (nueva_col queda NaN) para detectar
    productos nuevos que el mapeo todavia no cubre, o SUBGRUPO que se
    excluyen a proposito (ver comentario de CATEGORIAS_POR_SUBGRUPO['REX'])
    -- solo dentro de las filas de la propia linea (col_linea_filtro): antes
    de filtrar_linea, df trae subgrupos de OTRAS lineas mezclados que nunca
    van a matchear a proposito, y listarlos ahi seria puro ruido.
    """
    df = df.copy()
    diccionario = CATEGORIAS_POR_SUBGRUPO.get(linea)
    if diccionario is None:
        raise KeyError(f'No hay mapeo de categorias para LINEA={linea!r}. '
                        f'Lineas disponibles: {list(CATEGORIAS_POR_SUBGRUPO)}')
    df[nueva_col] = df[col_subgrupo].map(diccionario)

    en_linea = df[col_linea_filtro] == linea if col_linea_filtro in df.columns else pd.Series(True, index=df.index)
    sin_match = sorted(df.loc[en_linea & df[nueva_col].isna(), col_subgrupo].unique())
    if sin_match:
        print(f'⚠️  {len(sin_match)} SUBGRUPO de LINEA_FIN={linea!r} sin match en CATEGORIAS: {sin_match}')

    return df
