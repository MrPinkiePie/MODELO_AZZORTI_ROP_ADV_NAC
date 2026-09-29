# Cómo usar esta plantilla para un cliente nuevo

Este proyecto pronostica la demanda de un país ("destino") usando como
anticipo la demanda de otro país ("origen"), desfasada un número fijo de
campañas — la idea es que lo que se vende en el país origen hoy "anticipa"
lo que se va a vender en el país destino unas campañas después. Está armado
como plantilla: se puede copiar para armar el modelo de otro par de países,
línea o cliente sin reescribir todo el código desde cero.

## Qué hay en el proyecto

```
MODELO_AZZORTI_ROP_ADV_NAC/
├── config_proyecto.py   ← parámetros comunes (países, desfase, campañas, corte, corrida)
├── 0.- MOD_RF/         ← modelo por REFERENCIA (CAMP_NOMBRE_SAP)
└── 0.- MOD_TLL/         ← modelo por TALLA (CODI_VENT): reparte el de referencia
```

`0.- MOD_RF` y `0.- MOD_TLL` tienen la misma estructura cookiecutter y los
mismos nombres de archivos y funciones. Cada una tiene su propio `3.- src/`.
Es el mismo esquema de `MODELO_AZZORTI_ROP_NAC_NAC`.

## Paso 1: Copiar la carpeta y ponerle un nombre nuevo

No trabajes directo sobre esta carpeta. Copiala entera y ponele el nombre
del nuevo proyecto (por ejemplo `MODELO <NUEVO_CLIENTE>`). Recién ahí abrí
una sesión de trabajo sobre la copia — no es seguro renombrar la carpeta de
un proyecto que está en uso.

## Paso 2: Cambiar el nombre de proyecto y de modelo

Cada carpeta tiene su propio `config.py` con el nombre del modelo, que queda
pegado al `.pkl` entrenado:

| Archivo | Valor actual |
|---|---|
| `0.- MOD_RF/3.- src/config.py` | `NOMBRE_MODELO = "ROP_ADV_NAC"` |
| `0.- MOD_TLL/3.- src/config.py` | `NOMBRE_MODELO = "ROP_ADV_NAC_TALLAS"` |

`NOMBRE_PROYECTO = "AZZORTI"` va igual en los dos.

## Paso 3: Editar `config_proyecto.py` (raíz): países, desfase, campañas y corte

Es el **único** lugar donde se definen los parámetros que tienen que
coincidir entre el modelo de referencia y el de tallas. Los dos `config.py`
los importan y re-exportan, así que **no se editan** en los `entrenar.py`:

```python
PAIS_DESTINO = 'GUA'      # pais que se quiere pronosticar: per col bol gua ecu
PAIS_ORIGEN = 'COL'       # pais que anticipa al destino: per y col
LINEA = 'REX'             # linea de producto: soporta rin y rex
DESFASE_CAMPANAS = 3      # cuantas campanas antes se ve el pais origen

CAMP_INI = 202201
CAMP_FIN = 202613         # subirlo cuando haya campañas nuevas
CORTE_TRAIN_TEST = 202516 # >= esta campaña queda como TEST

COMENTARIO_REG = 'PRUEBA 1'  # nombre de la corrida de MOD_RF (lo lee MOD_TLL)
```

Los factores regionales por país de origen (`FACTORES_REGIONALES_POR_PAIS`)
siguen en `0.- MOD_RF/3.- src/config.py`, porque solo los usa la referencia.

## Paso 4: Revisar de dónde vienen los datos

Si el cliente nuevo usa las mismas carpetas compartidas de datos de la
empresa (OneDrive), no hay que tocar nada más. Si usa carpetas distintas,
hay que actualizar las rutas en ese mismo archivo de configuración — es el
único lugar donde viven.

## Paso 5: Revisar la lógica propia del negocio

Qué se filtra, cómo se arman las categorías de producto y la configuración
del modelo siguen siendo específicos de este par de países. Usalo como
ejemplo de cómo está armado, pero revisalo con el equipo antes de darlo por
bueno para un cliente distinto — esta parte no se adapta sola con cambiar
un par de nombres.

## Paso 6: Correr los modelos (en este orden)

1. `0.- MOD_RF/3.- src/models/entrenar.py`: entrena el modelo por
   referencia y deja los dos insumos del modelo de tallas:
   - `0.- data/interim/dat.parquet` (PASO 4): la base por talla ya limpia y
     cruzada ADV/NAC, antes de agrupar a referencia. Es el equivalente al
     `dat_12.parquet` de los scripts de tallas originales de ECU.
   - `0.- data/processed/PRUEBAS DE MODELO/df_{PAIS_ORIGEN}_{PAIS_DESTINO}_{LINEA}_{COMENTARIO_REG}.csv`
     (PASO 15): el pronóstico final por referencia.
2. `0.- MOD_TLL/3.- src/models/entrenar.py`: **solo carga** esos dos
   archivos (`RUTA_BASE_TALLAS` y `RUTA_PRED_REFERENCIA` en su `config.py`),
   reparte los percentiles por talla y cierra con la comparación SVS/SBR
   referencia vs tallas.

Si cambiás `COMENTARIO_REG` o algo de `config_proyecto.py`, primero volvé a
correr MOD_RF.

## Flags de cada `entrenar.py`

Son los mismos en MOD_RF y MOD_TLL (y en `MODELO_AZZORTI_ROP_NAC_NAC`):

```python
usar_boruta = False  # True reduce variables con Boruta antes de Optuna
max_iter = 20        # iteraciones de Boruta
usar_optuna = False  # False usa los hiperparámetros fijos; True corre Optuna
n_trials = 500       # trials de Optuna (en MOD_TLL arranca en 20)
guardar_pkl = True   # guarda el pkl entrenado en 4.- models/ (reemplaza el anterior)
```

Las variables fijas (`FEATURES_FIJAS`) están al inicio de cada `entrenar.py`,
justo debajo de los flags. Las candidatas para Boruta están en
`construir_features.py`: `obtener_features_candidatas` en MOD_RF y
`FEATURES_CANDIDATAS_TALLAS` en MOD_TLL.

## Modelo por tallas: `0.- MOD_TLL`

Replica los scripts `ECU_ADV_NAC_RIN_MODELO_TALLAS.py` y
`BOL_ADV_NAC_REX_MODELO_TALLAS.py` dentro de la estructura estándar:

- Un RandomForest predice la participación de cada talla dentro de su
  referencia en el NAC (`PART_NAC`) a partir de su participación en el ADV
  (`PART_ADV`), el volumen ADV de la referencia (`SUM_FACT_ADV`,
  `FACT_ADV_RGN`), el número de tallas, `SUBGRUPO` y `TALLA_NEW`.
- La predicción se rebalancea para que cada referencia sume 1 y se multiplica
  por los percentiles de MOD_RF. La suma de las tallas devuelve exactamente
  el pronóstico de la referencia.
- El PASO 13 compara SVS/SBR de REFERENCIA vs TALLAS_RF vs TALLAS_BASELINE
  (repartir igual que el ADV, sin modelo). Deja un gráfico en
  `5.- reports/figures/` y un Excel en `5.- reports/excel/`.
- **Solo vale la métrica de TEST.** En TRAIN los percentiles de MOD_RF son de
  dentro de muestra, así que salen optimistas.
- La base por talla solo trae tallas con venta NAC y ADV mayor a 0 (así la
  filtra MOD_RF), igual que la data de los scripts originales.

## Nota: todavía falta la parte de producción

A diferencia de `MODELO_AZZORTI_HOG_NAC_NAC` 
este proyecto todavía no tiene una carpeta de producción (`1.- MOD_RF_1`).


Acá el flujo completo (cargar datos, construir el modelo y evaluarlo) vive
en `0.- MOD_RF/3.- src/models/entrenar.py`. Ver el `README.md` de este
proyecto para más detalle sobre lo que falta.

## Dónde quedan las corridas y cómo compararlas en Power BI

| Carpeta `PRUEBAS DE MODELO` | Qué guarda | Para qué |
|---|---|---|
| `0.- MOD_RF/0.- data/processed/` | Solo corridas de **referencia** | Historial del modelo de referencia |
| `0.- MOD_TLL/0.- data/processed/` | Corridas de **referencia y de talla** | Comparar las dos en el Power BI (`DSB.pbix`) |

- `0.- MOD_RF/.../entrenar.py` guarda su csv en su carpeta **y una copia en la
  de MOD_TLL**, solo si el proyecto tiene `0.- MOD_TLL`.
- `0.- MOD_TLL/.../entrenar.py` guarda su csv **solo en su carpeta**.
- Cada csv trae la columna **`NIVEL`** (`REFERENCIA` o `TALLA`) además de
  `COMENTARIO`, para filtrar en el Power BI.
- `consolidar_pruebas_modelo.py` (en cada carpeta) junta todos los `df_*` en
  `consolidado_pruebas_modelo.parquet`, que se reemplaza en cada corrida.
  Agrega `ARCHIVO_ORIGEN`, `NIVEL` (la deduce del nombre si un csv viejo no la
  trae) y `ES_COMUN`, que marca las referencias presentes en todos los
  archivos. Filtrá `ES_COMUN = True` para comparar corridas sobre el mismo
  universo de referencias.
- Los csv se **reemplazan** si repetís el mismo `comentario`; para guardar una
  corrida nueva, cambiá el comentario (`COMENTARIO_REG` en `config_proyecto.py`).
- El consolidador también arma el **formato del DSB** (el mismo `TST_REF_PREV`
  de los scripts de tallas ECU/BOL), sin tocar los `entrenar.py`: agrega
  `COMPANIA`, `LINEA`, `MODELO`, `VERSION` (= `COMENTARIO`),
  `CAMP_ORIGEN/DESTINO`, `CODI_VENT` (en referencia = `NOMBRE_SAP`),
  `DESCRIPCION`, `Q_50` (= `y_pred`), `Q_REAL` (= `y_real`), `Q_RG3`, `Q_MKT`,
  `Q_FALT`, `TOT_PED_*`, `SPLIT` (`TRAIN/VALIDACION` en ambos proyectos),
  `ORDEN_SPLIT` y `TIPO`. Las columnas que un modelo no calcula (ej.
  `QUINTIL_PREC_ADV`, o las variables de ADV en NAC_NAC) quedan **vacías pero
  presentes**, porque Power Query falla si falta alguna columna.
