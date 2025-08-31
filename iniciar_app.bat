@echo off
TITLE Analizador de Performance v21 SINGLE BUTTON - Gemini

echo.
echo =================================================================
echo      INICIANDO APLICACION DE ANALISIS DE PERFORMANCE v21
echo      SINGLE BUTTON SYSTEM - UN SOLO BOTON PARA TODO  
echo =================================================================
echo.
echo ✅ CARACTERÍSTICAS v21 - UN SOLO BOTÓN:
echo    🚀 BOTÓN ÚNICO: Filtrar + Exportar + Guardar TODO
echo    🧠 Auto-ajuste inteligente de valores fuera de rango
echo    📊 Workflow completo automatizado en un solo clic
echo.

REM ** PASO 1: NAVEGAR A LA CARPETA CORRECTA **
cd "C:\Users\JuanCruz\Desktop_Local\SCRIPTS UTILIZADOS ORDENAR\gemini app analizador performance"

echo.
echo ** Verificando/Instalando librerias (pandas, polars, etc.)... **
REM Se reinstala pandas y se mantiene polars para un enfoque hibrido y robusto.
pip install streamlit plotly numpy pandas polars pyarrow > nul

echo.
echo ** Lanzando la aplicacion Streamlit v20 Enhanced... **
echo ** URL: http://localhost:8051 **
echo.
streamlit run app.py

echo.
pause
