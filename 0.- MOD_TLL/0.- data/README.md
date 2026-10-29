# 0.- data

- **raw/** — Datos originales tal como llegan. No se editan ni sobrescriben nunca.
- **interim/** — Datos con limpieza parcial, todavia no listos para modelar.
- **processed/** — Datos finales a nivel talla (`df_limpio.parquet`, con PART_NAC/PART_ADV y split TRAIN/TEST), listos para entrenar.
