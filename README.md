# 🏭 OPTIMITIVE OPTIBAT ANALYTICS SUITE

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 📊 Descripción

Suite profesional de análisis de rendimiento OPTIBAT desarrollada para **Optimitive**. Esta aplicación web permite el análisis avanzado de datos industriales con capacidades de visualización interactiva y generación de reportes HTML profesionales.

## ✨ Características Principales

### 🎯 Análisis Dual Independiente
- **Análisis de Eficiencia General**: Distribución de estados del sistema con solo filtros de fechas
- **Análisis Detallado de KPIs**: Comparación ON vs OFF con filtro OPTIBAT_READY=1

### 📈 Visualizaciones Avanzadas
- **Serie Temporal Interactiva**: Múltiples variables con sombreado por estados
- **Distribuciones KDE**: Análisis estadístico con líneas de densidad suavizadas
- **Boxplots Comparativos**: Análisis ON vs OFF con detección de outliers
- **Donut Charts**: Métricas de eficiencia con rangos de fechas automáticos

### 🔧 Funcionalidades Técnicas
- **Multi-DataFrame**: Selección independiente de datos por análisis
- **Filtros Globales**: Rango de fechas aplicable a toda la suite
- **Exportación HTML**: Reportes interactivos con Plotly embebido
- **Soporte Multi-Formato**: OSF, TXT, CSV, Excel (.xlsx/.xls)

### 🛡️ Seguridad
- **Acceso Restringido**: Sistema de autenticación empresarial
- **Interfaz Profesional**: Diseño corporativo Optimitive

## 🚀 Uso en Streamlit Cloud

### Acceso Directo
1. Visita: [https://your-app-url.streamlit.app](https://your-app-url.streamlit.app)
2. Credenciales de acceso:
   - **Usuario**: OPTIMITIVE
   - **Contraseña**: Mantenimiento.optibat

### Flujo de Trabajo
1. **Carga de Datos**: Subir archivos de estadísticas industriales
2. **Selección de DataFrames**: Elegir datos específicos para cada análisis
3. **Configuración de Filtros**: Ajustar rangos de fechas globales
4. **Análisis Multi-KPI**: Seleccionar variables para análisis simultáneo
5. **Exportación**: Generar reportes HTML profesionales

## 📋 Estructura del Proyecto

```
gemini-app-analizador-performance/
├── app.py                 # 🎯 Aplicación principal
├── requirements.txt       # 📦 Dependencias
├── README.md             # 📖 Documentación
└── iniciar_app.bat       # 🚀 Launcher local (Windows)
```

## 🛠️ Instalación Local (Opcional)

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/gemini-app-analizador-performance.git
cd gemini-app-analizador-performance

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app.py
```

## 📊 Casos de Uso

### 🏭 Análisis de Producción Industrial
- Monitoreo de eficiencia operacional
- Identificación de tiempos desperdiciados
- Análisis de variables críticas de proceso

### 📈 Reportes Ejecutivos
- Generación automática de reportes HTML
- Visualizaciones interactivas para presentaciones
- Métricas de rendimiento con filtros temporales

### 🔬 Análisis Técnico Avanzado
- Comparación estadística ON vs OFF
- Distribuciones de variables con KDE
- Análisis multi-variable simultáneo

## 🎨 Tecnologías

- **Frontend**: Streamlit + Plotly + HTML/CSS
- **Backend**: Python + Pandas + Polars
- **Análisis**: NumPy + SciPy (KDE)
- **Exportación**: Plotly.js embebido
- **Formatos**: Excel, CSV, TSV, OSF

## 👨‍💻 Desarrollador

**Desarrollado por JC Erreguerena** para **Optimitive**

---

### 🌟 Características Destacadas

- ✅ **Autenticación Empresarial**
- ✅ **Análisis Multi-DataFrame Independiente** 
- ✅ **Exportación HTML con Plotly Interactivo**
- ✅ **Filtro OPTIBAT_READY=1 para Análisis Operacional**
- ✅ **Visualizaciones Científicas con KDE**
- ✅ **Interfaz Profesional Optimitive**
- ✅ **Soporte Archivos Grandes (500MB+)**
- ✅ **Responsivo y Mobile-Friendly**

---
*Suite de Análisis OPTIBAT - Versión Profesional para Optimitive*