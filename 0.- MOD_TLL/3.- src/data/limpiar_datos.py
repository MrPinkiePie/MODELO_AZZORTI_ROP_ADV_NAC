"""
Funciones de limpieza sobre las tablas 1.1/1.2 ya cargadas (ver cargar_datos.py):
cruce con vehiculo/estrategia, marcado de atipicos/faltantes, y filtros de
rango/negocio. Migradas desde "0.- MOD_0.py" (la logica de destino/ECU y
origen/COL estaba duplicada ahi; aqui queda una sola version parametrizada
por zona/pais).
"""
import numpy as np
import pandas as pd


def cruzar_vehiculo_estrategia(df, df_veh_est, vehiculos_validos):
    """
    Cruza con la tabla de VEHICULO/ESTRATEGIA (ver cargar_datos.load_vehiculo_estrategia)
    por CAMP_CODI_VENT y se queda solo con las filas cuyo VEHICULO esta en
    vehiculos_validos (ej. ['NAL','NALADV'] para destino, ['ADV','NALADV','CATALG'] para origen).
    """
    df = pd.merge(df, df_veh_est[['CAMP_CODI_VENT', 'VEHICULO', 'ESTRATEGIA']],
                   on='CAMP_CODI_VENT', how='left')
    df = df[df['VEHICULO'].isin(vehiculos_validos)].copy()
    print(f'🧹 FILTRO VEHICULO {vehiculos_validos}: {len(df)} FILAS')
    return df


def marcar_atipicos(df, df_inflados, volar_atipicos=True):
    """
    Cruza df con la tabla de inflados (ver cargar_datos.load_inflados, ya trae
    CAMP_CODI_VENT) y marca PROB_ATIPICO_NEW ('SI'/'NO') a nivel CAMP+NOMBRE_SAP:
    si alguna talla (CODI_VENT) del producto esta inflada, se marca el producto
    completo. Si volar_atipicos=True, elimina las filas marcadas 'SI'.
    """
    df = df.copy()
    df = pd.merge(df, df_inflados[['CAMP_CODI_VENT', 'PROB_ATIPICO']], on='CAMP_CODI_VENT', how='left')
    df['PROB_ATIPICO'] = np.where(df['PROB_ATIPICO'].isna(), 'NO', 'SI')

    df['CAMP_NOMBRE_SAP'] = df['CAMP'].astype(str) + '_' + df['NOMBRE_SAP'].astype(str)
    hay_atipico_en_grupo = (df['PROB_ATIPICO'] == 'SI').groupby(df['CAMP_NOMBRE_SAP']).transform('any')
    df['PROB_ATIPICO_NEW'] = np.where(hay_atipico_en_grupo, 'SI', 'NO')

    print(f'====== CRUCE CON ATIPICOS: {len(df)} FILAS ======')
    print(df['PROB_ATIPICO_NEW'].value_counts())

    if volar_atipicos:
        df = df[df['PROB_ATIPICO_NEW'] == 'NO'].copy()
        print(f'🧹 FILTRO VOLAR ATIPICOS: {len(df)} FILAS')
    return df


def marcar_faltantes(df, volar_faltantes=True):
    """
    Crea FALT_NEW ('SI'/'NO') a partir de CORTES_CON_FALT, a nivel CAMP+NOMBRE_SAP.
    Si volar_faltantes=True, elimina las filas marcadas 'SI'.
    """
    df = df.copy()
    df['CORTES_CON_FALT'] = df['CORTES_CON_FALT'].apply(lambda x: 'NO' if pd.isna(x) else 'SI')
    df['CAMP_NOMBRE_SAP'] = df['CAMP'].astype(str) + '_' + df['NOMBRE_SAP'].astype(str)
    hay_falt_en_grupo = (df['CORTES_CON_FALT'] == 'SI').groupby(df['CAMP_NOMBRE_SAP']).transform('any')
    df['FALT_NEW'] = np.where(hay_falt_en_grupo, 'SI', 'NO')

    print(f'====== CRUCE CON FALTANTES: {len(df)} FILAS ======')
    print(df['FALT_NEW'].value_counts())

    if volar_faltantes:
        df = df[df['FALT_NEW'] == 'NO'].copy()
        print(f'🧹 FILTRO VOLAR FALTANTES: {len(df)} FILAS')
    return df


def filtrar_linea(df, linea, col_linea='LINEA_FIN'):
    """Filtra por linea de producto. Ver nota de orden en cargar_datos.load_1_1."""
    df = df[df[col_linea].isin([linea])].copy()
    print(f'🧹 FILTRO {col_linea}={linea}: {len(df)} FILAS')
    return df


def filtrar_pagina_valida(df, col_pagina, max_pag=300):
    """Filtra pagina > 0 y <= max_pag (paginas fuera de ese rango son datos invalidos)."""
    df = df[(df[col_pagina] > 0) & (df[col_pagina] <= max_pag)].copy()
    print(f'🧹 FILTRO {col_pagina} en (0, {max_pag}]: {len(df)} FILAS')
    return df


def filtrar_positivos(df, columnas):
    """Se queda con las filas donde TODAS las columnas dadas son > 0 (precio, cantidades, factores)."""
    mask = pd.Series(True, index=df.index)
    for col in columnas:
        mask &= df[col] > 0
    df = df[mask].copy()
    print(f'🧹 FILTRO {list(columnas)} > 0: {len(df)} FILAS')
    return df


def filtrar_surtido_sorpresa(df, col_nombre_sap='NOMBRE_SAP'):
    """Excluye productos SURTIDO/SORPRESA (no son un producto real a nivel referencia)."""
    mask = ~df[col_nombre_sap].str.contains(r'\b(SURT|SORPRESA)\b', regex=True)
    df = df[mask].copy()
    print(f'🧹 FILTRO SURT/SORPRESA: {len(df)} FILAS')
    return df


def marcar_posibles_atipicos_ratio(df, col_ratio, col_expo, nueva_col='FLAG_ATIPICO', volar_atipicos=True):
    """
    Adaptado de pos_atipicos() en "0.- MOD_0.py": marca 'POSIBLE_ATIPICO' cuando
    col_ratio supera el umbral (media + 3 desviaciones estandar) Y col_expo
    (ej. RATIO_EXPOSICION) es menor a 1 — un salto de factor que no se explica
    por mayor exposicion en pagina es sospechoso de ser un atipico.

    A diferencia de marcar_posibles_atipicos_ratio de MODELO_AZZORTI_ROP_NAC_NAC
    (que ademas exige un tercer filtro de ratio de precio > 1), el script
    original de este modelo no aplicaba ese filtro adicional.
    """
    df = df.copy()
    umbral = df[col_ratio].mean() + 3 * df[col_ratio].std()
    df[nueva_col] = np.where((df[col_ratio] > umbral) & (df[col_expo] < 1), 'POSIBLE_ATIPICO', 'NORMAL')

    print(f'====== {nueva_col}: {len(df)} FILAS ======')
    print(df[nueva_col].value_counts())

    if volar_atipicos:
        df = df[df[nueva_col] == 'NORMAL'].copy()
        print(f'🧹 FILTRO VOLAR {nueva_col}: {len(df)} FILAS')
    return df
