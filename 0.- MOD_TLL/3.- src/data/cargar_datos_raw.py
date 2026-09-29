"""
Trae los archivos de datos desde las rutas externas (ver config.py) y los
copia hacia 0.- data/raw/ con la fecha de extraccion, para tener una copia
local congelada y poder reproducir resultados aunque el archivo de origen
cambie despues.

No implementado todavia (el codigo original, "0.- MOD_0.py", leia siempre
directo de las rutas externas de OneDrive, sin copiar a data/raw/ primero).

Deberia contener:
- Una funcion que reciba la ruta de origen (externa) y la carpeta destino en
  data/raw/, y copie el archivo con un nombre que incluya la fecha de
  extraccion (ej. ventas_2026-08-24.csv).
- Un bloque `if __name__ == "__main__":` para poder ejecutar la copia de forma
  manual, sin que se dispare al importar el archivo desde otro script.
- Solo el copiado. Nada de limpieza ni transformacion de datos aqui.
"""
