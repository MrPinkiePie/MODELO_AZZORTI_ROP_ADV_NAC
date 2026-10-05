import os
import shutil
import subprocess
from pathlib import Path

# Configurar rutas
BASE_DIR = Path(r"C:\Users\mauricio.aguirre\35159_147728_DUPREE_VENTA_DIRECTA_S_A\STATS - Documentos\6.- TIM\29.- MAU\4.- ADV_GUA\MODELO_AZZORTI_ROP_ADV_NAC")
CONFIG_PATH = BASE_DIR / "config_proyecto.py"
RESULTADOS_DIR = BASE_DIR / "Resultados_Madrugada"

# Scripts a correr
RF_SCRIPT = BASE_DIR / "0.- MOD_RF" / "3.- src" / "models" / "entrenar.py"
TLL_SCRIPT = BASE_DIR / "0.- MOD_TLL" / "3.- src" / "models" / "entrenar.py"

# Carpetas de origen de los reportes
RF_REPORTS = BASE_DIR / "0.- MOD_RF" / "5.- reports"
TLL_REPORTS = BASE_DIR / "0.- MOD_TLL" / "5.- reports"
RF_PRUEBAS = BASE_DIR / "0.- MOD_RF" / "0.- data" / "processed" / "PRUEBAS DE MODELO"
TLL_PRUEBAS = BASE_DIR / "0.- MOD_TLL" / "0.- data" / "processed" / "PRUEBAS DE MODELO"

def set_decay_and_comment(status_bool, comentario):
    """Edita config_proyecto.py y entrenar.py de RF"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        lineas = f.readlines()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        for linea in lineas:
            if linea.strip().startswith("COMENTARIO_REG"):
                f.write(f"COMENTARIO_REG = '{comentario}'\n")
            else:
                f.write(linea)
    print(f"\n[+] config_proyecto.py actualizado: COMENTARIO_REG = '{comentario}'")

    with open(RF_SCRIPT, "r", encoding="utf-8") as f:
        lineas_rf = f.readlines()
    with open(RF_SCRIPT, "w", encoding="utf-8") as f:
        for linea in lineas_rf:
            if linea.strip().startswith("usar_ponderacion_temporal ="):
                f.write(f"usar_ponderacion_temporal = {status_bool} # Modificado por batch\n")
            else:
                f.write(linea)
    print(f"[+] entrenar.py (RF) actualizado: usar_ponderacion_temporal = {status_bool}")


def correr_script_y_capturar_log(script_path, log_path):
    print(f"[+] Ejecutando {script_path.name}... (esto tomara tiempo)")
    
    # Ruta absoluta al Python del entorno virtual (env_ml)
    python_exe = r"C:\Users\mauricio.aguirre\OneDrive - 35159_147728_DUPREE_VENTA_DIRECTA_S_A\Documentos\Mauricio\env_ml\Scripts\python.exe"
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    with open(log_path, "w", encoding="utf-8") as f_out:
        subprocess.run([python_exe, str(script_path)], cwd=str(BASE_DIR), stdout=f_out, stderr=subprocess.STDOUT, check=True, env=env)
    print(f"[+] {script_path.name} termino. Consola guardada en {log_path.name}")

def guardar_resultados(origen_reports, origen_pruebas, nombre_carpeta_destino, log_path=None):
    destino = RESULTADOS_DIR / nombre_carpeta_destino
    destino.mkdir(parents=True, exist_ok=True)
    
    # 1. Copiar Reports
    if origen_reports.exists():
        for item in origen_reports.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(origen_reports)
                (destino / "reports" / rel_path.parent).mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, destino / "reports" / rel_path)
                
    # 2. Copiar Pruebas de Modelo
    if origen_pruebas.exists():
        for item in origen_pruebas.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(origen_pruebas)
                (destino / "pruebas_de_modelo" / rel_path.parent).mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, destino / "pruebas_de_modelo" / rel_path)
                
    # 3. Mover Log
    if log_path and log_path.exists():
        shutil.move(str(log_path), str(destino / log_path.name))
        
    print(f"[+] Archivos copiados a {destino.name}")

def main():
    RESULTADOS_DIR.mkdir(exist_ok=True)
    
    print("\n" + "="*50)
    print("INICIANDO FASE 1: CON DECAY (PRUEBA 6)")
    print("="*50)
    
    # 1. RF (¡SALTADO! El usuario ya lo está corriendo)
    print("[+] Saltando ejecución de RF con Decay porque ya fue ejecutado por el usuario.")
    guardar_resultados(RF_REPORTS, RF_PRUEBAS, "RF_Con_Decay") 
    
    # 2. TALLAS
    log_tll = BASE_DIR / "consola_TLL_con_decay.txt"
    # Ya ejecutado, saltamos: correr_script_y_capturar_log(TLL_SCRIPT, log_tll)
    guardar_resultados(TLL_REPORTS, TLL_PRUEBAS, "TLL_Con_Decay", log_tll)
    
    print("\n" + "="*50)
    print("INICIANDO FASE 2: SIN DECAY (PRUEBA 7)")
    print("="*50)
    set_decay_and_comment(False, "PRUEBA 7")
    
    # 3. RF
    log_rf2 = BASE_DIR / "consola_RF_sin_decay.txt"
    correr_script_y_capturar_log(RF_SCRIPT, log_rf2)
    guardar_resultados(RF_REPORTS, RF_PRUEBAS, "RF_Sin_Decay", log_rf2)
    
    # 4. TALLAS
    log_tll2 = BASE_DIR / "consola_TLL_sin_decay.txt"
    correr_script_y_capturar_log(TLL_SCRIPT, log_tll2)
    guardar_resultados(TLL_REPORTS, TLL_PRUEBAS, "TLL_Sin_Decay", log_tll2)
    
    print("\n" + "="*50)
    print("¡TODOS LOS PROCESOS TERMINADOS CON ÉXITO!")
    print("="*50)

if __name__ == "__main__":
    main()
