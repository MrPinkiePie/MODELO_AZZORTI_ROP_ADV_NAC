"""
Script interactivo para analizar la volumetría de datos de Entrenamiento vs Prueba (Train/Test)
según diferentes escenarios de campañas de inicio.
Útil para la toma de decisiones sobre qué CAMP_INI_MODELO usar sin tener que re-entrenar el modelo.
"""
# %%
import sys
from pathlib import Path
import pandas as pd

# Asegurar importes desde src
sys.path.append(str(Path(__file__).resolve().parents[1] / "3.- src"))
from config import DATA_PROCESSED, CORTE_TRAIN_TEST, REPORTS_EXCEL

def calcular_volumetria():
    print("Cargando histórico completo de datos (df_limpio)...")
    try:
        df = pd.read_parquet(DATA_PROCESSED / 'df_limpio.parquet')
    except FileNotFoundError:
        print("ERROR: No se encontró df_limpio.parquet. Asegúrate de correr entrenar.py primero con el CAMP_INI más antiguo.")
        return

    # Escenarios de inicio que queremos simular
    escenarios_inicio = [202201, 202301, 202401]
    
    resultados = []

    for camp_ini in escenarios_inicio:
        # Filtrar solo el periodo de interés (desde este camp_ini simulado)
        df_escenario = df[df['CAMP'] >= camp_ini]
        
        # Separar virtualmente Train y Test
        mask_train = df_escenario['CAMP'] < CORTE_TRAIN_TEST
        mask_test = df_escenario['CAMP'] >= CORTE_TRAIN_TEST
        
        df_train = df_escenario[mask_train]
        df_test = df_escenario[mask_test]
        
        filas_train = len(df_train)
        filas_test = len(df_test)
        total_filas = filas_train + filas_test
        
        # Calcular los rangos de campañas realmente encontrados
        rango_train = f"{df_train['CAMP'].min()} a {df_train['CAMP'].max()}" if filas_train > 0 else "Sin datos"
        rango_test = f"{df_test['CAMP'].min()} a {df_test['CAMP'].max()}" if filas_test > 0 else "Sin datos"

        # Ratio: Qué porcentaje de los datos cae en Train
        pct_train = (filas_train / total_filas) * 100 if total_filas > 0 else 0

        # Variación respecto al escenario anterior
        perdida_filas = 0
        pct_perdida = 0.0
        if len(resultados) > 0:
            filas_anteriores = resultados[-1]['Total Filas (Muestra)']
            perdida_filas = filas_anteriores - total_filas
            pct_perdida = (perdida_filas / filas_anteriores) * 100 if filas_anteriores > 0 else 0

        resultados.append({
            'Inicio Simulado': camp_ini,
            'Total Filas (Muestra)': total_filas,
            'Variación Filas': f"-{perdida_filas}" if perdida_filas > 0 else "-",
            '% Pérdida vs Anterior': f"-{pct_perdida:.1f}%" if pct_perdida > 0 else "-",
            '% Data en Train': f"{pct_train:.1f}%",
            'Rango Train': rango_train,
            'Filas Train': filas_train,
            'Rango Test': rango_test,
            'Filas Test': filas_test
        })

    # Crear dataframe de resultados
    df_res = pd.DataFrame(resultados)
    
    # Imprimir bonito en consola
    print("\n" + "="*95)
    print("REPORTE DE VOLUMETRÍA DE DATOS (TRAIN VS TEST)")
    print(f"Corte Oficial de Train/Test: {CORTE_TRAIN_TEST}")
    print("="*95)
    print(df_res.to_string(index=False))
    print("="*95)

    # Exportar a Excel
    ruta_excel = REPORTS_EXCEL / 'Volumetria_Train_Test.xlsx'
    
    with pd.ExcelWriter(ruta_excel, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, sheet_name='Escenarios_Volumetria', index=False)
        
        # Dar algo de formato a las columnas para que se lea fácil
        workbook = writer.book
        worksheet = writer.sheets['Escenarios_Volumetria']
        formato_numeros = workbook.add_format({'num_format': '#,##0'})
        formato_centrado = workbook.add_format({'align': 'center'})
        
        worksheet.set_column('A:A', 15)
        worksheet.set_column('B:B', 20, formato_numeros)
        worksheet.set_column('C:D', 18, formato_centrado)
        worksheet.set_column('E:E', 15, formato_centrado)
        worksheet.set_column('F:F', 20)
        worksheet.set_column('G:G', 15, formato_numeros)
        worksheet.set_column('H:H', 20)
        worksheet.set_column('I:I', 15, formato_numeros)

    print(f"\nReporte exportado exitosamente a: {ruta_excel.name}")

if __name__ == '__main__':
    calcular_volumetria()
