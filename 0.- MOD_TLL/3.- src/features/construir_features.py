"""
Construye las variables (features) de ROP_ADV_NAC a partir de las tablas ya
cargadas/limpiadas (ver data/cargar_datos.py y data/limpiar_datos.py).
Migradas desde "0.- MOD_0.py", donde el bloque de features de competencia
estaba duplicado (una vez para destino/ECU, otra para origen/COL) — aqui
queda una sola version parametrizada por sufijo.
"""
import sys
from pathlib import Path  # manipular rutas de forma OOP

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.mis_funciones import desfasar_camp


def obtener_features_candidatas(factores_regionales):
    """
    Set COMPLETO de variables candidatas (antes de Boruta). Unico lugar donde
    se define: lo usan entrenar.py (usar_boruta=True) y 2.- notebooks/2.-
    features.py (ranking de informacion mutua). Es funcion y no constante
    porque los factores regionales crudos cambian segun PAIS_ORIGEN (ver
    FACTORES_REGIONALES_POR_PAIS en config.py).
    """
    return {
        'target': 'LOG_RATIO_FACTOR',
        'num': [
            # numericas
            'RATIO_PRECIO', 'FACT_ADV_RGN',
            # factores regionales (crudos, distintos por pais)
            'FACTOR_MAX', 'FACTOR_MIN', 'FACTOR_MEDIANA', *factores_regionales,
            # competidores origen
            'PRECIO_PROMEDIO_COMPETIDORES_ORIGEN', 'PRECIO_PROMEDIO_SUBGRUPO_ORIGEN',
            'RATIO_PROMEDIO_COMPETIDORES_PRECIO_ORIGEN', 'RATIO_PROMEDIO_PRECIO_ORIGEN_SUBGRUPO',
            'NUM_PRODUCTOS_POR_CATEGORIAS_ORIGEN', 'NUM_PRODUCTOS_POR_SUBGRUPO_ORIGEN',
            'NUM_PRODUCTOS_POR_CATEGORIAS_PAGINA_ORIGEN',
            # competidores destino
            'PRECIO_PROMEDIO_COMPETIDORES_DESTINO', 'PRECIO_PROMEDIO_SUBGRUPO_DESTINO',
            'RATIO_PROMEDIO_COMPETIDORES_PRECIO_DESTINO', 'RATIO_PROMEDIO_PRECIO_DESTINO_SUBGRUPO',
            'NUM_PRODUCTOS_POR_CATEGORIAS_DESTINO', 'NUM_PRODUCTOS_POR_SUBGRUPO_DESTINO',
            'NUM_PRODUCTOS_POR_CATEGORIAS_PAGINA_DESTINO',
            # precios
            'PRECIO_N', 'PRECIO_ADV_IN',
            # paginas
            'PAG_ADV_IN', 'PAG_NAC_NEW', 'PAG_ADV_IN_PAG_DER', 'PAG_NAC_NEW_PAG_DER', 'RATIO_PAG_NAC_ADV',
            # exposicion
            'EXPOSICION_ORIGEN', 'EXPOSICION_DESTINO', 'RATIO_EXPOSICION',
            # temporales
            'CAMP_SEN', 'CAMP_COS',
        ],
        'cat': ['SUBGRUPO', 'CATEGORIAS', 'GRUPO', 'CAMBIO_PAG_LADO', 'TIPO_REF',
                # mezcla origen/destino (mismo patron que MODELO_AZZORTI_ROP_NAC_NAC)
                'ESTR_MIX', 'UBICACION_PAG_MIX', 'NUM_CAMP_MIX'],
    }




def construir_features_competencia(df, col_camp, col_pagina, col_precio, col_linea, sufijo, cols_extra_pagina=None):
    """
    Requiere que df ya tenga CATEGORIAS (ver cargar_datos.mapear_categorias_por_subgrupo)
    y SUBGRUPO. sufijo: 'DESTINO' u 'ORIGEN'.
    cols_extra_pagina: columnas adicionales para el agrupado de pagina (en el
    script original, destino agrupaba tambien por PAIS y origen no, aunque en
    origen es redundante porque solo hay un pais).
    """
    df = df.copy()
    cols_pagina = [col_camp] + (cols_extra_pagina or []) + [col_pagina, 'CATEGORIAS']

    df[f'NUM_PRODUCTOS_POR_CATEGORIAS_PAGINA_{sufijo}'] = df.groupby(cols_pagina)['NOMBRE_SAP'].transform('nunique')
    df[f'NUM_PRODUCTOS_POR_CATEGORIAS_{sufijo}'] = df.groupby([col_camp, col_linea, 'CATEGORIAS'])['NOMBRE_SAP'].transform('nunique')
    df[f'NUM_PRODUCTOS_POR_SUBGRUPO_{sufijo}'] = df.groupby([col_camp, col_linea, 'SUBGRUPO'])['NOMBRE_SAP'].transform('nunique')
    df[f'PRECIO_PROMEDIO_COMPETIDORES_{sufijo}'] = df.groupby([col_camp, 'CATEGORIAS'])[col_precio].transform('mean')
    df[f'PRECIO_PROMEDIO_SUBGRUPO_{sufijo}'] = df.groupby([col_camp, 'SUBGRUPO'])[col_precio].transform('mean')
    df[f'RATIO_PROMEDIO_COMPETIDORES_PRECIO_{sufijo}'] = df[col_precio] / df[f'PRECIO_PROMEDIO_COMPETIDORES_{sufijo}']
    df[f'RATIO_PROMEDIO_PRECIO_{sufijo}_SUBGRUPO'] = df[col_precio] / df[f'PRECIO_PROMEDIO_SUBGRUPO_{sufijo}']
    return df


def calcular_camp_mas_desfase(df, col_camp, desfase_campanas=3):
    """
    Agrega CAMP_MAS_3: la campana de destino a la que corresponde esta fila
    de origen, adelantando desfase_campanas campanias. NO toca CAMP_CODI_VENT
    (para poder seguir cruzando por la clave de origen sin desfasar hasta que
    haga falta -- ver reconstruir_clave_desfasada).
    """
    df = df.copy()
    df[f'CAMP_MAS_{desfase_campanas}'] = df[col_camp].astype(int).apply(
        lambda c: desfasar_camp(c, retroceso=-desfase_campanas))
    return df


def reconstruir_clave_desfasada(df, col_camp_mas, col_codi_vent='CODI_VENT'):
    """
    Reconstruye CAMP_CODI_VENT usando la campana YA desfasada (col_camp_mas,
    ver calcular_camp_mas_desfase), para cruzar esta fila de origen con la
    campana equivalente de destino via cruzar_origen_destino().
    """
    df = df.copy()
    df['CAMP_CODI_VENT'] = df[col_camp_mas].astype(str) + '_' + df[col_codi_vent].astype(str)
    return df


def cruzar_origen_destino(df_destino, df_origen_con_clave, col_clave='CAMP_CODI_VENT'):
    """Une destino y origen ya alineados por campana (ver construir_clave_cruce_origen)."""
    df = pd.merge(df_destino, df_origen_con_clave, on=col_clave, how='inner')
    print(f'====== CRUCE ORIGEN/DESTINO: {len(df)} FILAS ======')
    return df


def calcular_factor_nacional(df, col_pedidos='Q_PEDI_N', col_total='TOT_N', col_mkt='Q_MKT_N', col_total_mkt='TOT_PED_MKT_NAC'):
    """FAC_NAC: proporcion de pedidos nacionales sobre el total. FAC_MKT: idem para marketing."""
    df = df.copy()
    df['FAC_NAC'] = df[col_pedidos] / df[col_total]
    df['FAC_MKT'] = df[col_mkt] / df[col_total_mkt]
    return df


def pag_der(df, col):
    """Marca si la pagina es del lado derecho (par) o izquierdo (impar) del catalogo."""
    df = df.copy()
    df[f'{col}_PAG_DER'] = np.where(df[col] % 2 == 0, 0, 1)
    return df


def calcular_cambio_pagina_lado(df, col_pagina_origen_der, col_pagina_destino_der):
    """
    Compara el lado de pagina (par/impar) entre origen y destino: 'NO CAMBIO',
    'DER_A_IZQ', 'IZQ_A_DER' (o 'REVISAR' si ninguna condicion aplica).
    """
    df = df.copy()
    condiciones = [
        df[col_pagina_origen_der] == df[col_pagina_destino_der],
        df[col_pagina_origen_der] > df[col_pagina_destino_der],
        df[col_pagina_origen_der] < df[col_pagina_destino_der],
    ]
    valores = ['NO CAMBIO', 'DER_A_IZQ', 'IZQ_A_DER']
    df['CAMBIO_PAG_LADO'] = np.select(condiciones, valores, default='REVISAR')
    return df


def calcular_ubicacion_pagina(df, col, nueva_col, umbral=100):
    """
    INICIALES/FINALES segun el corte de pagina umbral (mismo patron que
    UBICACION_PAG_ORIGEN/DESTINO de MODELO_AZZORTI_ROP_NAC_NAC). A diferencia
    de CAMBIO_PAG_LADO (par/impar, lado izq/der de un mismo pliego), esto
    marca la mitad del catalogo (portada vs. contraportada) -- son dos
    dimensiones distintas de una misma pagina, no redundantes entre si.
    """
    df = df.copy()
    df[nueva_col] = np.where(df[col] <= umbral, 'INICIALES', 'FINALES')
    return df


def calcular_ratio_precio(df, col_destino='PRECIO_N', col_origen='PRECIO_ADV_IN'):
    """RATIO_PRECIO: cuanto mas caro/barato es el producto en destino vs. origen."""
    df = df.copy()
    df['RATIO_PRECIO'] = df[col_destino] / df[col_origen]
    return df


def agrupar_por_referencia(df, columnas_suma, col_grupo='CAMP_NOMBRE_SAP'):
    """
    Colapsa filas a nivel de referencia (CAMP_NOMBRE_SAP), sumando columnas_suma
    (factores/cantidades ya calculados por talla). Ver "Modelo Referencias" en
    0.- MOD_0.py. Los factores agregados solo llegan a nivel referencia (SAP),
    no a nivel talla -- ver README.md de este modelo.
    """
    resultado = df.groupby(col_grupo, as_index=False)[columnas_suma].sum()
    print(f'====== AGRUPANDO POR REFERENCIA ({col_grupo}): {len(df)} FILAS -> {len(resultado)} FILAS ======')
    return resultado


def unir_extras_por_referencia(df_agregado, df_extras, col_grupo='CAMP_NOMBRE_SAP'):
    """Vuelve a pegar las columnas descriptivas (precio, grupo, subgrupo, etc.) que no se suman."""
    extras_unicos = df_extras.drop_duplicates(subset=col_grupo)
    return pd.merge(df_agregado, extras_unicos, how='inner', on=col_grupo)


def calcular_ratio_factor(df, col_destino='FAC_NAC', col_origen='FACT_ADV_RGN'):
    """
    RATIO_FACTOR: base del target del modelo (se entrena sobre su logaritmo,
    LOG_RATIO_FACTOR -- ver entrenar.py). Cuanto de la demanda de destino se
    explica por la de origen. Mismo nombre que RATIO_FACTOR de
    MODELO_AZZORTI_ROP_NAC_NAC (alli ademas suma un log-shift +0.0001*1000;
    aca se deja en escala simple, igual que SBR_SVR_RF.py).
    """
    df = df.copy()
    df['RATIO_FACTOR'] = df[col_destino] / df[col_origen]
    return df


def calcular_resumen_factores_regionales(df, factores):
    """
    Agrega FACTOR_MAX/MIN/MEDIANA y REGION_MAX/MIN por fila (df ya viene a
    nivel referencia, una fila por CAMP_NOMBRE_SAP), a partir de las columnas
    de 'factores' (factores regionales de Colombia: FACT_BOG, FACT_OCC1,
    FACT_SAN1, FACT_COS). El minimo evita regiones en cero cuando es posible,
    para no elegir como "region minima" una region sin datos.
    """
    def _resumen_fila(fila):
        valores = fila[factores]
        valores_no_cero = valores[valores > 0]
        if not valores_no_cero.empty:
            min_valor = valores_no_cero.min()
            region_min = valores_no_cero.idxmin()
        else:
            min_valor = valores.min()
            region_min = valores.idxmin()
        return pd.Series({
            'FACTOR_MAX': valores.max(), 'REGION_MAX': valores.idxmax(),
            'FACTOR_MIN': min_valor, 'REGION_MIN': region_min,
            'FACTOR_MEDIANA': valores.median(),
        })

    resumen = df.apply(_resumen_fila, axis=1)
    return pd.concat([df.reset_index(drop=True), resumen.reset_index(drop=True)], axis=1)


def marcar_tipo_referencia(df, col_producto='NOMBRE_SAP', col_camp='CAMP'):
    """PRIMERA_CAMP + TIPO_REF ('NUEVA' si es la primera campana del producto, si no 'CON_HISTORIA')."""
    df = df.sort_values(col_camp).copy()
    primera_camp = (df.groupby(col_producto)[col_camp].min()
                     .reset_index().rename(columns={col_camp: 'PRIMERA_CAMP'}))
    df = pd.merge(df, primera_camp, on=col_producto, how='left')
    df['TIPO_REF'] = np.where(df[col_camp] == df['PRIMERA_CAMP'], 'NUEVA', 'CON_HISTORIA')
    return df


def calcular_exposicion(df):
    """
    EXPOSICION_x: inverso de cuantos productos compiten en la misma
    pagina+categoria (mas productos = menos exposicion por producto).
    RATIO_EXPOSICION: si el producto gano o perdio exposicion relativa al
    pasar de origen a destino.
    """
    df = df.copy()
    df['EXPOSICION_ORIGEN'] = 1 / df['NUM_PRODUCTOS_POR_CATEGORIAS_PAGINA_ORIGEN']
    df['EXPOSICION_DESTINO'] = 1 / df['NUM_PRODUCTOS_POR_CATEGORIAS_PAGINA_DESTINO']
    df['RATIO_EXPOSICION'] = df['EXPOSICION_DESTINO'] / df['EXPOSICION_ORIGEN'].replace(0, np.nan)
    return df


def calcular_variables_ciclicas_campana(df, col_num_camp, n_camps=18):
    """Codificacion ciclica (seno/coseno) del numero de campana dentro del anio."""
    df = df.copy()
    df['CAMP_SEN'] = np.sin(2 * np.pi * df[col_num_camp].astype(int) / n_camps)
    df['CAMP_COS'] = np.cos(2 * np.pi * df[col_num_camp].astype(int) / n_camps)
    return df


# ===========================================================================
# MODELO POR TALLAS (propio de 0.- MOD_TLL)
# Reparte el pronostico de referencia de 0.- MOD_RF entre sus tallas, con la
# misma logica de los scripts ECU_ADV_NAC_RIN / BOL_ADV_NAC_REX_MODELO_TALLAS:
# la participacion de la talla en el ADV (PART_ADV) explica su participacion
# en el NAC (PART_NAC).
# ===========================================================================

# Set completo de variables candidatas del modelo de tallas (antes de Boruta).
# Mismas del script BOL_ADV_NAC_REX (N_SKU de alla = NUM_TALLAS aca).
# SUM_FACT_NAC no entra: es el real de la referencia en destino (leakage).
FEATURES_CANDIDATAS_TALLAS = {
    'target': 'PART_NAC',
    'num': ['PART_ADV', 'SUM_FACT_ADV', 'FACT_ADV_RGN', 'NUM_TALLAS', 'HHI_TALLAS', 'DESVIACION_TALLA_LAG_3'],
    'cat': ['SUBGRUPO', 'TALLA_NEW'],
}


def calcular_participacion_talla(df):
    """
    Participacion de cada talla (fila de dat.parquet) dentro de su referencia
    (CAMP_NOMBRE_SAP), en destino (NAC) y en origen (ADV):
      PART_NAC = FAC_NAC / SUM_FACT_NAC   (target)
      PART_ADV = FACT_ADV_RGN / SUM_FACT_ADV
    Agrega NUM_TALLAS (tallas de la referencia en esa campaña). Como MOD_RF
    ya filtro FAC_NAC y FACT_ADV_RGN > 0, las sumas nunca son 0.
    """
    df = df.copy()
    df['TALLA_NEW'] = df['TALLA_NEW'].astype(str).str.strip()

    por_referencia = df.groupby('CAMP_NOMBRE_SAP')
    df['SUM_FACT_ADV'] = por_referencia['FACT_ADV_RGN'].transform('sum')
    df['SUM_FACT_NAC'] = por_referencia['FAC_NAC'].transform('sum')
    df['NUM_TALLAS'] = por_referencia['FAC_NAC'].transform('size')

    df['PART_NAC'] = df['FAC_NAC'] / df['SUM_FACT_NAC']
    df['PART_ADV'] = df['FACT_ADV_RGN'] / df['SUM_FACT_ADV']

    print(f'====== PARTICIPACION POR TALLA CREADA: {len(df)} FILAS, '
          f'{df["CAMP_NOMBRE_SAP"].nunique()} REFERENCIAS ======')
    return df


def calcular_hhi_tallas(df):
    """
    Calcula el HHI de la distribución de tallas para cada referencia en el país de origen (ADV).
    HHI_TALLAS alto = demanda concentrada en 1 o 2 tallas.
    HHI_TALLAS bajo = demanda distribuida uniformemente entre todas las tallas.
    """
    df = df.copy()
    df['PART_ADV_SQ'] = df['PART_ADV'] ** 2
    df['HHI_TALLAS'] = df.groupby('CAMP_NOMBRE_SAP')['PART_ADV_SQ'].transform('sum')
    df = df.drop(columns=['PART_ADV_SQ'])
    return df


def calcular_desviacion_historica_talla(df, col_camp='CAMP', col_producto='NOMBRE_SAP', col_talla='TALLA_NEW', col_target='PART_ADV', ventanas=(3,)):
    """
    Calcula la participación histórica promedio de una talla para un producto específico
    usando campañas anteriores (shift) para detectar desviaciones en la campaña actual.
    """
    df = df.copy()
    
    # 1. Agrupar por producto, talla y campana
    agg_hist = df.groupby([col_producto, col_talla, col_camp])[col_target].mean().reset_index()
    agg_hist = agg_hist.sort_values([col_producto, col_talla, col_camp])
    
    # 2. Desplazar (shift) y calcular promedios moviles por producto-talla
    for v in ventanas:
        col_name = f'PART_ADV_HIST_LAG_{v}'
        agg_hist[col_name] = agg_hist.groupby([col_producto, col_talla])[col_target].transform(
            lambda x: x.shift(1).rolling(window=v, min_periods=1).mean()
        )
        
    # 3. Cruzar de vuelta con el dataframe original
    cols_a_unir = [col_producto, col_talla, col_camp] + [f'PART_ADV_HIST_LAG_{v}' for v in ventanas]
    df = pd.merge(df, agg_hist[cols_a_unir], on=[col_producto, col_talla, col_camp], how='left')
    
    # 4. Calcular desviación actual contra la historia y rellenar nulos
    for v in ventanas:
        col_name = f'PART_ADV_HIST_LAG_{v}'
        col_desv = f'DESVIACION_TALLA_LAG_{v}'
        
        # Si no hay historia, asumimos que no hay desviación (la historia era exactamente lo que vendio hoy)
        df[col_name] = df[col_name].fillna(df[col_target])
        
        # La anomalía es: lo que vendió hoy - lo que vendía históricamente
        df[col_desv] = df[col_target] - df[col_name]
        
    return df


