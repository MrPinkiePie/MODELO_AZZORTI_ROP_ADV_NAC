#%%
"""
Funciones de EDA (exploracion), reutilizables entre notebooks de exploracion.
Copiado de MODELO_AZZORTI_ROP_NAC_NAC/0.- MOD_RF/3.- src/data/eda.py (mismo
patron, mismo default columna_factor='FACTOR_RCP' para no romper ese origen
si algun dia se comparan). En ADV_NAC el equivalente de FACTOR_RCP es
RATIO_FACTOR (dat_ref, ver PASO 5 de entrenar.py) -- pasar
columna_factor='RATIO_FACTOR' al llamar estas funciones.

Unica diferencia con el original: graficar_violin_factor_por_estrategia y
graficar_ranking_factor_por_estrategia ahora reciben 'columna_grupo' (antes
tenian 'ESTRATEGIA' fijo) porque dat_ref de ADV_NAC no trae ESTRATEGIA a nivel
referencia, sino ESTR_MIX (mezcla de estrategia origen+destino, ver PASO 4 de
entrenar.py).
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression


def resumen_nulos(df):
    """Porcentaje de nulos por columna, ordenado de mayor a menor."""
    print('====== RESUMEN DE NULOS POR COLUMNA ======')
    return print((df.isna().mean() * 100).sort_values(ascending=False))


def resumen_cardinalidad(df):
    """Cantidad de valores unicos por columna, para detectar columnas casi constantes."""
    print('====== RESUMEN DE CARDINALIDAD POR COLUMNA ======')
    return print(df.nunique().sort_values())


def crear_anio_camp(df, columna_camp='CAMP'):
    """Separa la campania (formato AAAACC, campanias 1-18) en anio y numero de campania."""
    print(f'====== CREANDO ANIO Y NUM_CAMPANA DESDE {columna_camp} ======')
    df = df.copy()
    camp = pd.to_numeric(df[columna_camp], errors='coerce').astype('Int64')  # CAMP llega como texto
    df['anio'] = (camp // 100)
    df['num_campana'] = (camp % 100)
    df['anio'] = df['anio'].astype(str)
    df['num_campana'] = df['num_campana'].astype(str)
    return df


def graficar_factor_rcp_por_campana(df, columna_factor='FACTOR_RCP', agregacion='mean', figsize=(16, 8)):
    """
    Factor_rcp por numero de campania (1-18), comparando entre anios.
    agregacion='mean' (promedio) o 'sum' (suma) de columna_factor por campania.
    figsize=(ancho, alto) en pulgadas, para ajustar el tamaño del grafico.
    """
    etiqueta = 'SUMA' if agregacion == 'sum' else 'PROMEDIO'
    print(f'====== GRAFICO DE {columna_factor} POR CAMPANA ({etiqueta}) ======')
    resumen = df.groupby(['anio', 'num_campana'])[columna_factor].agg(agregacion).reset_index()
    # num_campana es texto (ver crear_anio_camp), se ordena por su valor numerico (1 a 18)
    resumen = resumen.sort_values('num_campana', key=lambda col: col.astype(int))

    print(f'====== TABLA CAMPANA x ANIO ({etiqueta}) DE {columna_factor} ======')
    tabla = resumen.pivot(index='num_campana', columns='anio', values=columna_factor)
    tabla = tabla.reindex(sorted(tabla.index, key=int))
    print(tabla.to_string())

    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=resumen, x='num_campana', y=columna_factor, hue='anio', marker='o', sort=False, ax=ax)
    ax.set_xlabel('Campaña')
    ax.set_ylabel(f'{columna_factor} ({etiqueta})')
    return fig, ax


def graficar_factor_rcp_por_anio(df, columna_factor='FACTOR_RCP'):
    """Distribucion de factor_rcp por anio."""
    print(f'====== GRAFICO DE {columna_factor} POR ANIO ======')
    print(f'====== ESTADISTICAS DE {columna_factor} POR ANIO ======')
    print(df.groupby('anio')[columna_factor].describe().to_string())

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='anio', y=columna_factor, ax=ax)
    ax.set_xlabel('Año')
    ax.set_ylabel(columna_factor)
    return fig, ax


def graficar_histograma_factor_rcp(df, columna_factor='FACTOR_RCP', bins=10, top_n=5,
                                    columna_camp='CAMP', columna_descripcion='NOMBRE_SID_FACT'):
    """Histograma de factor_rcp, para ver su distribucion y posibles outliers.

    Traza una linea vertical en media + 3 desviaciones estandar como
    referencia visual del umbral tipico de outlier. Ademas imprime una tabla
    con los 'top_n' rangos con mas registros, y una tabla con los 'top_n'
    productos (camp, descripcion, factor) con mayor factor_rcp, ya que en el
    grafico es dificil distinguir cuantos registros caen en cada tramo.
    """
    print('====== GRAFICO DE HISTOGRAMA ======')
    media = df[columna_factor].mean()
    limite_3sd = media + 3 * df[columna_factor].std()

    n_outliers = (df[columna_factor] > limite_3sd).sum()
    pct_outliers = n_outliers / len(df) * 100
    print(f'====== PRODUCTOS QUE SUPERAN MEDIA + 3 DE ({limite_3sd:.4f}): {n_outliers} ({pct_outliers:.2f}% del total) ======')

    print(f'====== TABLA DE FRECUENCIAS (TOP {top_n} RANGOS) DE {columna_factor} ======')
    tabla_rangos = pd.cut(df[columna_factor], bins=bins, precision=4).value_counts()
    print(tabla_rangos)

    print(f'====== TOP {top_n} PRODUCTOS CON MAYOR {columna_factor} ======')
    top_productos = df.nlargest(top_n, columna_factor)[[columna_camp, columna_descripcion, columna_factor]]
    print(top_productos.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=df, x=columna_factor, kde=True, ax=ax)
    ax.axvline(limite_3sd, color='red', linestyle='--', label=f'Media + 3 DE = {limite_3sd:.4f}')
    ax.set_xlabel(columna_factor)
    ax.legend()
    return fig, ax


def filtrar_top_n_categorias(df, por, columna_factor='FACTOR_RCP', top_n=5):
    """
    Se queda solo con las 'top_n' categorias de la columna 'por' (ej. GRUPO,
    SUBGRUPO, CODI_VENT) con mayor factor_rcp promedio. Util para no saturar
    los graficos de subgrupos cuando 'por' tiene muchos valores unicos.
    """
    print(f'====== FILTRANDO TOP {top_n} DE {por} POR {columna_factor} PROMEDIO ======')
    top_categorias = df.groupby(por)[columna_factor].mean().nlargest(top_n).index
    return df[df[por].isin(top_categorias)]


def graficar_boxplot_factor_rcp(df, columna_factor='FACTOR_RCP', por=None, top_n=5):
    """
    Boxplot de factor_rcp, en agregado (por=None) o separado por una columna
    categorica (ej. por='anio', por='GRUPO', por='SUBGRUPO', por='CODI_VENT').
    Al analizar subgrupos (por != None), se limita a las 'top_n' categorias
    con mayor factor_rcp promedio (top_n=None desactiva el filtro).
    """
    print(f'====== GRAFICO DE BOXPLOT DE {columna_factor} POR {por} ======')
    if por is not None and top_n is not None:
        df = filtrar_top_n_categorias(df, por, columna_factor, top_n)

    print(f'====== ESTADISTICAS DE {columna_factor} POR {por} ======')
    if por is None:
        print(df[columna_factor].describe().to_string())
    else:
        print(df.groupby(por)[columna_factor].describe().to_string())

    fig, ax = plt.subplots()
    if por is None:
        sns.boxplot(data=df, y=columna_factor, ax=ax)
    else:
        sns.boxplot(data=df, x=por, y=columna_factor, ax=ax)
        ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel(columna_factor)
    return fig, ax


def cruzar_con_vehiculo_estrategia(df, df_veh_est, vehiculos=('NAL', 'NALADV')):
    """
    Cruza df (debe tener CAMP_CODI_VENT) con la tabla de vehiculo/estrategia
    (ver load_vehiculo_estrategia en cargar_datos.py) y se queda solo con los
    'vehiculos' indicados (por defecto NAL y NALADV). En ADV_NAC normalmente
    no hace falta: dat_ref ya trae ESTR_MIX (mezcla de estrategia
    origen+destino, ver PASO 4 de entrenar.py) sin necesidad de re-cruzar.
    """
    print(f'====== CRUCE CON VEHICULO/ESTRATEGIA (VEHICULOS={vehiculos}) ======')
    df = pd.merge(df, df_veh_est, on='CAMP_CODI_VENT', how='left')
    df = df[df['VEHICULO'].isin(vehiculos)].copy()
    print(f'====== FILAS TRAS FILTRO DE VEHICULO: {len(df)} ======')

    print('====== CANTIDAD DE REGISTROS POR ESTRATEGIA Y VEHICULO ======')
    tabla_conteo = pd.crosstab(df['ESTRATEGIA'], df['VEHICULO'])
    print(tabla_conteo.to_string())
    return df


def filtrar_top_n_por_frecuencia(df, por, top_n=5):
    """
    Se queda solo con las 'top_n' categorias de la columna 'por' con mas
    registros (frecuencia), a diferencia de filtrar_top_n_categorias que
    filtra por el promedio de una variable. Util cuando categorias con pocos
    registros (ej. n=2) distorsionan la comparacion aunque tengan un promedio
    alto.
    """
    print(f'====== FILTRANDO TOP {top_n} DE {por} POR CANTIDAD DE REGISTROS ======')
    top_categorias = df[por].value_counts().nlargest(top_n).index
    return df[df[por].isin(top_categorias)]


def graficar_violin_factor_por_estrategia(df, columna_factor='FACTOR_RCP', columna_grupo='ESTRATEGIA',
                                           top_n=8, figsize=(16, 8)):
    """
    Compara la distribucion de 'columna_factor' entre las categorias de
    'columna_grupo' con mas registros (top_n por cantidad de salidas, no por
    promedio del factor, para no comparar categorias con muestra chica).
    columna_grupo='ESTRATEGIA' en NAC_NAC/HOG_NAC_NAC; en ADV_NAC usar
    columna_grupo='ESTR_MIX' (dat_ref no trae ESTRATEGIA a nivel referencia).
    """
    print(f'====== VIOLIN DE {columna_factor} POR {columna_grupo} (TOP {top_n} POR CANTIDAD) ======')
    if top_n is not None:
        df = filtrar_top_n_por_frecuencia(df, columna_grupo, top_n)

    print(f'====== ESTADISTICAS DE {columna_factor} POR {columna_grupo} ======')
    print(df.groupby(columna_grupo)[columna_factor].describe().to_string())

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=df, x=columna_grupo, y=columna_factor, ax=ax)
    ax.set_xlabel(columna_grupo)
    ax.set_ylabel(columna_factor)
    ax.tick_params(axis='x', rotation=45)
    return fig, ax


def graficar_ranking_factor_por_estrategia(df, columna_factor='FACTOR_RCP', columna_grupo='ESTRATEGIA',
                                            top_n=8, figsize=(14, 7)):
    """
    Rankea las categorias de 'columna_grupo' con mas registros (top_n por
    cantidad de salidas) de mayor a menor promedio de 'columna_factor', con
    intervalo de confianza del 95%. A diferencia del violin (forma de la
    distribucion), esto responde directamente "cual categoria jala mas
    factor" en un solo vistazo: el intervalo ademas avisa cuando una
    categoria con pocos registros no es una estimacion confiable, aunque su
    promedio se vea alto. columna_grupo='ESTRATEGIA' en NAC_NAC/HOG_NAC_NAC;
    en ADV_NAC usar columna_grupo='ESTR_MIX'.
    """
    print(f'====== RANKING DE {columna_factor} POR {columna_grupo} (TOP {top_n} POR CANTIDAD) ======')
    if top_n is not None:
        df = filtrar_top_n_por_frecuencia(df, columna_grupo, top_n)

    resumen = df.groupby(columna_grupo)[columna_factor].agg(['count', 'mean', 'std'])
    resumen['ic_95'] = 1.96 * resumen['std'] / np.sqrt(resumen['count'])
    resumen = resumen.sort_values('mean', ascending=False)

    print(f'====== TABLA DE {columna_factor} POR {columna_grupo} (ORDENADA POR PROMEDIO) ======')
    print(resumen.to_string())

    fig, ax = plt.subplots(figsize=figsize)
    sns.pointplot(data=df, x=columna_grupo, y=columna_factor, order=resumen.index, errorbar=('ci', 95),
                  linestyle='none', ax=ax)
    ax.set_xlabel(columna_grupo)
    ax.set_ylabel(f'{columna_factor} (promedio + IC 95%)')
    ax.tick_params(axis='x', rotation=45)
    return fig, ax


def resumen_factor_por_nombre_sap(df, columna_factor='FACTOR_RCP', columna_anio='anio',
                                   columna_estrategia='ESTRATEGIA', top_n=20):
    """
    Resumen por NOMBRE_SAP: promedio de columna_factor por anio (una columna
    por anio, para ver evolucion del producto), mas promedio y desviacion
    estandar general, cantidad de salidas y cantidad de categorias distintas
    de 'columna_estrategia' en las que salio ese producto (en ADV_NAC usar
    columna_estrategia='ESTR_MIX').
    Requiere que df ya tenga 'anio' (ver crear_anio_camp) y 'columna_estrategia'.
    Limitado a los top_n productos con mas salidas (top_n=None desactiva el
    filtro), para no imprimir miles de filas.
    """
    print(f'====== RESUMEN DE {columna_factor} POR NOMBRE_SAP (TOP {top_n} POR CANTIDAD) ======')
    if top_n is not None:
        df = filtrar_top_n_por_frecuencia(df, 'NOMBRE_SAP', top_n)

    tabla_anio = df.pivot_table(index='NOMBRE_SAP', columns=columna_anio, values=columna_factor, aggfunc='mean')

    resumen_general = df.groupby('NOMBRE_SAP').agg(
        cantidad_salidas=(columna_factor, 'count'),
        promedio=(columna_factor, 'mean'),
        desviacion=(columna_factor, 'std'),
        cantidad_estrategias=(columna_estrategia, 'nunique'),
    )

    tabla = tabla_anio.join(resumen_general).sort_values('promedio', ascending=False)

    # se redondean solo las columnas de factor (anios + promedio/desviacion), no los conteos
    columnas_factor = list(tabla_anio.columns) + ['promedio', 'desviacion']
    tabla[columnas_factor] = tabla[columnas_factor].round(4)

    print(tabla.to_string())
    return tabla


def graficar_tendencia_factor_rcp(df, columna_factor='FACTOR_RCP', por=None, top_n=5, agregacion='mean', figsize=(16, 8)):
    """
    Factor_rcp por num_campana, para estudiar tendencias.
    En agregado (por=None) o separado por grupo (ej. por='anio', por='GRUPO',
    por='SUBGRUPO', por='CODI_VENT') - una linea por cada valor de 'por'.
    Al analizar subgrupos (por != None), se limita a las 'top_n' categorias
    con mayor factor_rcp promedio (top_n=None desactiva el filtro).
    agregacion='mean' (promedio) o 'sum' (suma) de columna_factor por campania.
    figsize=(ancho, alto) en pulgadas, para ajustar el tamaño del grafico.
    """
    etiqueta = 'SUMA' if agregacion == 'sum' else 'PROMEDIO'
    print(f'====== GRAFICO DE TENDENCIA DE {columna_factor} {por} {top_n} ({etiqueta}) ======')
    if por is not None and top_n is not None:
        df = filtrar_top_n_categorias(df, por, columna_factor, top_n)

    columnas_grupo = ['num_campana'] if por is None else ['num_campana', por]
    resumen = df.groupby(columnas_grupo)[columna_factor].agg(agregacion).reset_index()
    # num_campana es texto (ver crear_anio_camp): se ordena por su valor numerico (1 a 18)
    # y se usa sort=False en el lineplot para que no lo reordene alfabeticamente de nuevo
    resumen = resumen.sort_values('num_campana', key=lambda col: col.astype(int))

    print(f'====== TABLA DE TENDENCIA POR NUM_CAMPANA {por} ({etiqueta}) ======')
    if por is None:
        print(resumen.to_string(index=False))
    else:
        tabla = resumen.pivot(index='num_campana', columns=por, values=columna_factor)
        tabla = tabla.reindex(sorted(tabla.index, key=int))
        print(tabla.to_string())

    fig, ax = plt.subplots(figsize=figsize)
    if por is None:
        sns.lineplot(data=resumen, x='num_campana', y=columna_factor, marker='o', sort=False, ax=ax)
    else:
        sns.lineplot(data=resumen, x='num_campana', y=columna_factor, hue=por, marker='o', sort=False, ax=ax)
    ax.set_xlabel('Campaña')
    ax.set_ylabel(f'{columna_factor} ({etiqueta})')
    return fig, ax


def ranking_informacion_mutua(df, columna_target, candidatas_num, candidatas_cat, seed=42, figsize=(10, 8)):
    """
    Informacion mutua (MI) de cada variable candidata (numerica o categorica)
    contra columna_target: a diferencia de la correlacion, capta relaciones
    NO lineales sin necesitar entrenar un modelo completo (ver Kuhn & Johnson,
    Applied Predictive Modeling, Cap. 18 "Filter methods"; Zheng, Feature
    Engineering for ML, Cap. 3). 0 = independencia, mayor valor = mas
    dependencia entre la variable y columna_target.

    Las categoricas se codifican con pd.factorize (a codigos enteros) solo
    para poder pasarlas a mutual_info_regression, marcandolas como discretas
    via discrete_features -- la codificacion es arbitraria (no ordinal), pero
    mutual_info_regression no asume orden en las columnas discretas.

    Devuelve (tabla, fig, ax): tabla ordenada de mayor a menor y el grafico
    de barras correspondiente (hacer plt.show() en el notebook para verlo).
    """
    print(f'====== INFORMACION MUTUA DE {columna_target} CON VARIABLES CANDIDATAS ======')
    X_cat = df[candidatas_cat].apply(lambda col: pd.factorize(col)[0])
    X = pd.concat([df[candidatas_num], X_cat], axis=1)
    es_discreta = [False] * len(candidatas_num) + [True] * len(candidatas_cat)

    mi = mutual_info_regression(X, df[columna_target], discrete_features=es_discreta, random_state=seed)
    tabla = pd.Series(mi, index=X.columns, name='informacion_mutua').sort_values(ascending=False).to_frame()
    tabla.index.name = 'variable'
    print(tabla.to_string())

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=tabla.reset_index(), x='informacion_mutua', y='variable', color='tab:blue', ax=ax)
    ax.set_xlabel(f'Informacion mutua con {columna_target}')
    ax.set_ylabel('')
    return tabla, fig, ax
