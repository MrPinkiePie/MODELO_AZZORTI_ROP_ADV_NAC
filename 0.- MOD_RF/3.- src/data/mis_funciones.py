import os
import re
import inspect  # para sacar el numero de linea de quien llama a log()/resumen_filtro()
import linecache  # para leer la linea de codigo exacta que llamo a resumen_filas() y sacar el nombre de la tabla
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


def calcular_error_unidades(y_pred, y_real):
    return np.abs(y_pred - y_real)


def calcular_error_pct(y_pred, y_real):
    return np.abs(y_pred - y_real) / y_real * 100


def svs_sbr(q_est, q_real):
    svs = np.where(q_est > q_real, q_real, q_est)
    sbr = np.where(q_est > q_real, q_est - q_real, 0)
    svs_total = round((sum(svs) / sum(q_real)) * 100, 2)
    sbr_total = round((sum(sbr) / sum(q_real)) * 100, 2)
    return svs_total, sbr_total


def camp_lineal(camp_id):
    if pd.isna(camp_id):
        return np.nan
    year = int(camp_id) // 100
    camp = int(camp_id) % 100
    return (year * 18) + camp


def desfasar_camp(camp, retroceso=3):
    """
    Retrocede 'retroceso' campanias desde 'camp' (formato AAAACC, 18 campanias
    por anio), cruzando de anio cuando hace falta (ej. 202601 retrocediendo 3
    campanias = 202516). Con retroceso negativo, ADELANTA campanias (ej.
    desfasar_camp(camp, retroceso=-3) es el CAMP+3 que usa este modelo para
    cruzar el ADV de origen con el NAC de destino).
    Para una columna completa, usar con .apply: df['CAMP'].apply(desfasar_camp)
    """
    lista_campanas = [anio * 100 + num for anio in range(2017, 2051)
                       for num in range(1, 19)]

    idx = lista_campanas.index(int(camp)) - retroceso
    if idx < 0 or idx >= len(lista_campanas):
        raise ValueError(f'{camp} desfasando {retroceso} campanias cae fuera del rango '
                          f'de lista_campanas (2017-2050)')
    return lista_campanas[idx]


def metricas(nombre, y, y_pred):
    print(f"\n--- {nombre} ---")
    print(f"MAE:  {mean_absolute_error(y, y_pred):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
    print(f"R2:   {r2_score(y, y_pred):.4f}")


def limpiar_codigo(serie):
    """
    Normaliza codigos a texto: si el valor es numerico, quita decimales (123.0 -> '123');
    si no lo es (ej. 'K124718'), lo deja tal cual.
    """
    numerico = pd.to_numeric(serie, errors='coerce')
    numerico_int = numerico.astype('Int64')
    texto = numerico_int.astype(str)
    original_texto = serie.astype(str)
    limpio = texto.where(numerico.notna(), original_texto)
    return limpio.str.strip()


def descomponer_camp(df, col_camp='CAMP', sufix=''):
    """
    Descompone una columna de campana (ej. 202614) en ANIO ('2026'), IND_CAMP ('C_14') y NUM_CAMP ('14').
    """
    df = df.copy()
    df[col_camp] = limpiar_codigo(df[col_camp])
    df['ANIO' + sufix] = df[col_camp].str[0:4]
    num_camp = df[col_camp].str[4:]
    df['IND_CAMP' + sufix] = 'C_' + num_camp.astype(str)
    df['NUM_CAMP' + sufix] = num_camp
    return df


def svs_sbr_por_grupo(df, col_grupo, col_pred='Q_50', col_real='FACTOR_RCP_DESTINO'):
    """
    Calcula SVS/SBR (ver svs_sbr) por cada valor de col_grupo.
    """
    resultado = (
        df.reset_index(drop=True)
        .groupby(col_grupo)
        .apply(lambda g: pd.Series(svs_sbr(g[col_pred], g[col_real]), index=['SVS', 'SBR']), include_groups=False)
        .reset_index()
    )
    return resultado


def log(mensaje):
    """
    print con el numero de linea de quien lo llamo adelante, ej. '[L142] registros...'
    Sirve para ubicar rapido en el archivo de donde salio cada print del log.
    """
    linea = inspect.currentframe().f_back.f_lineno
    print(f"[L{linea}] {mensaje}")


def titulo_paso(mensaje, ancho=70):
    """
    Imprime un encabezado separador (linea de '=' arriba y abajo del
    mensaje) para marcar claramente el inicio de un paso del pipeline. Se
    usa al arrancar cada bloque de entrenar.py, antes de sus resumen_filas().
    """
    separador = '=' * ancho
    print(f'\n{separador}\n{mensaje}\n{separador}')


def resumen_filas(df, mensaje, antes=None, umbral_alerta=0.5):
    """
    Imprime cuantas filas y columnas quedan tras un paso, con el numero de
    linea de quien lo llamo y el nombre de la tabla (variable) resumida --
    se saca leyendo la linea de codigo del llamado, ej. resumen_filas(df_x, ...)
    imprime 'df_x'. Si se pasa antes (DataFrame o int con el largo previo),
    agrega cuantas filas se ganaron/perdieron y el % retenido, y avisa si se
    retuvo menos de umbral_alerta.
    """
    frame = inspect.currentframe().f_back
    linea = frame.f_lineno
    codigo_llamada = linecache.getline(frame.f_code.co_filename, linea)
    match_tabla = re.search(r'resumen_filas\(\s*([A-Za-z_]\w*)', codigo_llamada)
    nombre_tabla = match_tabla.group(1) if match_tabla else 'tabla'

    texto = f"[L{linea}] 👉 {nombre_tabla}: {mensaje}: {len(df):,} filas x {df.shape[1]} columnas"
    if antes is not None:
        n_antes = antes if isinstance(antes, int) else len(antes)
        pct = len(df) / n_antes if n_antes else 0
        texto += f" ({len(df) - n_antes:+,} | {pct:.0%} retenido)"
        if pct < umbral_alerta:
            texto += f"  ⚠️ AVISO: retuvo menos del {umbral_alerta:.0%}"
    print(texto)
