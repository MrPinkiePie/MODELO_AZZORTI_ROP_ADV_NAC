# -*- coding: utf-8 -*-
"""
Concatena los .txt que exporta cada corrida de RF NUBE.py en esta misma
carpeta, en DATA.csv/DATA.parquet -- insumo del dashboard DASH NUBES RF.pbix.
Identico al de MODELO_AZZORTI_ROP_NAC_NAC (no depende de las features de un
modelo en particular).
"""

import pandas as pd
import os, glob
import numpy as np
from pathlib import Path

path = Path(__file__).resolve().parent
ftz = glob.glob(str(path / '*.txt'))
ftz = [f for f in ftz if 'DUPE' or 'DUCO' or 'LAPTOP' not in f]
ALL_FIN_2 = pd.concat([pd.read_csv(f) for f in ftz], ignore_index=True)


columnas_clave = ['MODELO', 'TARGET', 'VARIABLES']

# Crear el identificador de grupo
ALL_FIN_2['ANDROIDE'] = ALL_FIN_2.groupby(columnas_clave).ngroup()

ALL_FIN_2['ANDROIDE'] = ALL_FIN_2['ANDROIDE'].astype(str)
ALL_FIN_2['LEYENDA'] = ALL_FIN_2.MODELO.astype(str) + ': ' + ALL_FIN_2.TARGET.astype(str)


# Variables del combo con mas features (el mas completo) -- se usan como base
# para las columnas indicadoras has_<variable>
predictoras = ALL_FIN_2[ALL_FIN_2['NÚMERO DE VARIABLES'] == max(ALL_FIN_2['NÚMERO DE VARIABLES'])]
predictoras = predictoras['VARIABLES'].unique()
predictoras = predictoras.tolist()
import ast

predictoras = list(ast.literal_eval(predictoras[0]))

# Paso 1: Convertimos la columna 'VARIABLES' de string a lista (una sola vez)
ALL_FIN_2['VARIABLES_LIST'] = ALL_FIN_2['VARIABLES'].apply(ast.literal_eval)

# Paso 2: Creamos las columnas indicadoras de manera eficiente
for variable in predictoras:
    ALL_FIN_2[f'has_{variable}'] = ALL_FIN_2['VARIABLES_LIST'].apply(lambda x: "SI" if variable in x else "NO")

# Paso 3: Eliminamos la columna 'VARIABLES_LIST'
ALL_FIN_2.drop('VARIABLES_LIST', axis=1, inplace=True)


ALL_FIN_2.to_csv(path / 'DATA.csv', index=False)

ALL_FIN_2.to_parquet(path / 'DATA.parquet', index=False)
