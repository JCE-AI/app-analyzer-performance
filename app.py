import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import io
import glob
import os
import re
import base64
import plotly.io as pio

# --- Configuraciones Iniciales ---
st.set_page_config(
    page_title="Optimitive OPTIBAT Analytics Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings("ignore")

# =========================
# OPTIMITIVE THEME COLORS
# =========================
OPTIMITIVE_COLORS = {
    'primary_red': '#E31E32',
    'primary_black': '#000000',
    'dark_bg': '#FFFFFF',
    'medium_bg': '#F8F9FA',
    'light_bg': '#FFFFFF',
    'accent_blue': '#0099CC',
    'text_primary': '#2C3E50',
    'text_secondary': '#6C757D',
    'success': '#28A745',
    'warning': '#FFC107',
    'error': '#DC3545',
    'border': '#DEE2E6'
}

# ==============================================================================
# FUNCIONES DE UTILIDAD
# ==============================================================================

def to01(s):
    """Convierte series a binario 0/1"""
    if hasattr(s, 'to_pandas'):
        # Es Polars Series
        s_pandas = s.to_pandas()
    else:
        # Es Pandas Series
        s_pandas = s
    return pd.to_numeric(s_pandas, errors='coerce').fillna(0).clip(0, 1).astype(int)

# ==============================================================================
# FUNCIONES DE FILTRADO DE DATAFRAMES
# ==============================================================================

def get_available_dataframes() -> Dict[str, pd.DataFrame]:
    """Obtiene DataFrames disponibles del session state."""
    available = {}
    
    # DataFrame principal cargado
    if st.session_state.loaded_data and st.session_state.loaded_data.df is not None:
        df_main = st.session_state.loaded_data.df.to_pandas()
        available["df_principal"] = df_main
    
    # DataFrame filtrado si existe
    if st.session_state.df_filtrado is not None:
        available["df_filtrado"] = st.session_state.df_filtrado
    
    # Otros DataFrames almacenados
    for key, value in st.session_state.available_dataframes.items():
        if isinstance(value, pd.DataFrame) and not value.empty:
            available[key] = value
    
    return available

def apply_range_filters(df: pd.DataFrame, filter_ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Aplica filtros de rango a columnas numéricas."""
    if not filter_ranges:
        return df.copy()
    
    mask = pd.Series(True, index=df.index)
    for col, (min_val, max_val) in filter_ranges.items():
        if col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            mask &= (numeric_col >= min_val) & (numeric_col <= max_val)
    
    return df[mask.fillna(False)]

def apply_text_search(df: pd.DataFrame, search_text: str, search_cols: List[str], 
                     use_regex: bool = False, case_sensitive: bool = False) -> pd.DataFrame:
    """Aplica búsqueda de texto en columnas especificadas."""
    search_text = (search_text or "").strip()
    if not search_text:
        return df
    
    search_cols = [c for c in search_cols if c in df.columns]
    if not search_cols:
        return df
    
    mask = pd.Series(False, index=df.index)
    for col in search_cols:
        try:
            col_mask = df[col].astype(str).str.contains(
                search_text, regex=use_regex, case=case_sensitive, na=False
            )
            mask |= col_mask
        except:
            # Si hay error en regex, buscar como texto literal
            col_mask = df[col].astype(str).str.contains(
                re.escape(search_text), case=case_sensitive, na=False
            )
            mask |= col_mask
    
    return df[mask]

def clean_dataframe_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el DataFrame para exportación según criterio del script de referencia."""
    # Crear copia profunda para no modificar el original
    df_out = df.copy(deep=True)
    
    # ELIMINAR la columna 'datetime' duplicada si existe (solo para exportación)
    if 'datetime' in df_out.columns:
        # Buscar la columna de tiempo original
        time_cols = [c for c in df_out.columns if c.lower() in ['date', 'fecha', 'timestamp', 'time'] and c != 'datetime']
        if time_cols:
            # Si existe columna de tiempo original, eliminar la duplicada 'datetime'
            df_out = df_out.drop('datetime', axis=1)
    
    # IMPORTANTE: Mantener el orden original de las columnas
    # No reordenar - las columnas deben quedar como vienen del archivo fuente
    
    # Reemplazar cadenas vacías o solo espacios con np.nan
    df_out = df_out.replace(r'^\s*$', np.nan, regex=True)
    
    # Patrón para detectar variaciones de null/nan/none/nat (case-insensitive)
    pat_null = re.compile(r'^\s*(?:nan|null|none|nat)\s*$', flags=re.IGNORECASE)
    
    # Aplicar limpieza solo a columnas de tipo objeto
    obj_cols = df_out.select_dtypes(include='object').columns
    if len(obj_cols) > 0:
        df_out[obj_cols] = df_out[obj_cols].replace({pat_null: np.nan}, regex=True)
    else:
        # Si no hay columnas objeto, aplicar a todo el DataFrame
        df_out = df_out.replace({pat_null: np.nan}, regex=True)
    
    # Mantener el DataFrame con el mismo orden de columnas
    return df_out

# ==============================================================================
# 1. LÓGICA DE PROCESAMIENTO DE DATOS (v17 - ENFOQUE HÍBRIDO Y CACHÉ CORREGIDO)
# ==============================================================================

@dataclass
class LoadedData:
    df: pl.DataFrame
    file_count: int
    total_mb: float

@st.cache_data
def discover_data_files(data_directory: str) -> List[str]:
    """Descubre archivos de datos disponibles en el directorio."""
    if not os.path.exists(data_directory):
        return []
    
    file_patterns = ["*.txt", "*.tsv", "*.csv"]
    files = []
    
    for pattern in file_patterns:
        matched_files = glob.glob(os.path.join(data_directory, pattern))
        # Filtrar solo archivos que parecen ser de estadísticas
        for file_path in matched_files:
            filename = os.path.basename(file_path)
            if filename.startswith('2025-') and 'STATISTICS' in filename:
                files.append(file_path)
    
    return sorted(files)

@st.cache_data
def load_data_from_file(file_path: str) -> Optional[LoadedData]:
    """Carga datos desde un archivo específico."""
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content_str = f.read()
        
        filename = os.path.basename(file_path)
        df_single = process_file_content(content_str, filename)
        
        if df_single is not None and not df_single.is_empty():
            prepared_df = prepare_final_dataframe(df_single)
            if not prepared_df.is_empty():
                file_size = os.path.getsize(file_path)
                return LoadedData(
                    df=prepared_df, 
                    file_count=1, 
                    total_mb=file_size / (1024 * 1024)
                )
    except Exception as e:
        st.error(f"Error cargando archivo {filename}: {e}")
    
    return None

def make_columns_unique(columns: List[str]) -> List[str]:
    seen, new_columns = set(), []
    for col in columns:
        new_col, i = col, 1
        while new_col in seen:
            new_col = f"{col}_{i}"; i += 1
        new_columns.append(new_col); seen.add(new_col)
    return new_columns

@st.cache_data
def process_file_content(content_str: str, filename: str) -> Optional[pl.DataFrame]:
    """Usa Pandas para leer el contenido de un string (robusto y cacheable) y luego convierte a Polars."""
    try:
        lines = content_str.splitlines()
        if len(lines) < 11: return None

        header_list = [h.strip() for h in lines[1].split('\t')]
        unique_headers = make_columns_unique(header_list)
        
        # Usar PANDAS para la lectura, que es más tolerante
        df_pd = pd.read_csv(io.StringIO("\n".join(lines[10:])), sep='\t', header=None, names=unique_headers, engine='python')
        if df_pd.empty: return None

        # Convertir a Polars y forzar todo a texto para una unión segura
        df_pl = pl.from_pandas(df_pd)
        return df_pl.with_columns(pl.all().cast(pl.Utf8))
    except Exception as e:
        st.error(f"Error crítico al procesar contenido de {filename}: {e}")
        return None

def prepare_final_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    time_col_name = next((c for c in df.columns if c.lower() in ["date", "fecha", "datetime", "timestamp", "time"]), None)
    if not time_col_name:
        st.error("Error Crítico: No se pudo encontrar una columna de tiempo."); return pl.DataFrame()

    # PRESERVAR EL ORDEN ORIGINAL DE COLUMNAS
    original_columns = df.columns.copy()
    
    # Convertir la columna de tiempo MANTENIENDO SU NOMBRE ORIGINAL
    expressions = [pl.col(time_col_name).str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", strict=False)]
    
    # Procesar otras columnas
    for col_name in df.columns:
        if col_name != time_col_name:
            expressions.append(pl.col(col_name).cast(pl.Float64, strict=False))

    df_typed = df.with_columns(expressions)

    # Aplicar conversiones especiales manteniendo nombres originales
    final_expressions = []
    if "OPTIBAT_ON" in df_typed.columns: 
        final_expressions.append(pl.col("OPTIBAT_ON").cast(pl.Int32, strict=False).fill_null(0))
    if "OPTIBAT_READY" in df_typed.columns: 
        final_expressions.append(pl.col("OPTIBAT_READY").cast(pl.Int32, strict=False).fill_null(0))

    if final_expressions:
        df_typed = df_typed.with_columns(final_expressions)
    
    # Crear columna 'datetime' para uso interno PERO mantener el orden original
    df_with_datetime = df_typed.with_columns([
        pl.col(time_col_name).alias("datetime")
    ])
    
    # Seleccionar columnas en el orden original + datetime al final para uso interno
    final_df = df_with_datetime.select(original_columns + ["datetime"])
    
    return final_df.drop_nulls("datetime").sort("datetime")

def load_data(uploaded_files: List[Any]) -> Optional[LoadedData]:
    progress_bar = st.sidebar.progress(0, text="Iniciando carga...")
    dfs, total_size = [], 0
    for i, file in enumerate(uploaded_files):
        progress_text = f"({i+1}/{len(uploaded_files)}) Cargando: {file.name}"
        progress_bar.progress((i + 1) / len(uploaded_files), text=progress_text)
        
        # Leer contenido fuera de la función cacheada
        content_str = file.getvalue().decode("utf-8", errors="latin-1")
        df_single = process_file_content(content_str, file.name)
        
        if df_single is not None and not df_single.is_empty():
            dfs.append(df_single); total_size += file.size

    if not dfs:
        progress_bar.empty(); st.warning("No se pudieron cargar datos válidos."); return None

    progress_bar.progress(1.0, text="Combinando y preparando datos...")
    full_df = pl.concat(dfs, how='vertical')
    prepared_df = prepare_final_dataframe(full_df)
    progress_bar.empty()

    if prepared_df.is_empty():
        st.error("Falló la preparación de los datos."); return None
        
    return LoadedData(df=prepared_df, file_count=len(uploaded_files), total_mb=total_size / (1024 * 1024))

# ... (El resto de las funciones se mantienen sin cambios) ...
class ColorScheme:
    PRIMARY_GREEN = "#2E8B57"
    WARNING_ORANGE = "#FF7F50"
    CRITICAL_RED = "#DC143C"
    NEUTRAL_BLUE = "#4682B4"

    # 🔥 Colores de sombreado más intensos por estado OPTIBAT:
    GREEN_FILL = "rgba(16,185,129,0.35)"    # ON & READY  → verde más intenso
    ORANGE_FILL = "rgba(255,102,0,0.50)"    # OFF & READY → naranja más intenso  
    RED_FILL = "rgba(239,68,68,0.22)"       # OFF & NO READY (ligeramente más notorio)

    PALETTE = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#10b981","#ef4444","#f59e0b","#6366f1","#0ea5e9",
        "#22c55e","#f472b6","#84cc16","#14b8a6","#f97316"
    ]

def calculate_metrics(df: pl.DataFrame) -> Dict[str, Any]:
    n = len(df); on_ready=0; off_ready=0; off_nready=0
    if n > 0 and "OPTIBAT_ON" in df.columns and "OPTIBAT_READY" in df.columns:
        on_ready = df.filter((pl.col('OPTIBAT_ON') == 1) & (pl.col('OPTIBAT_READY') == 1)).height
        off_ready = df.filter((pl.col('OPTIBAT_ON') == 0) & (pl.col('OPTIBAT_READY') == 1)).height
        off_nready = df.filter(pl.col('OPTIBAT_READY') == 0).height
    return {
        "total_records": n, "date_min": df['datetime'].min() if n>0 else None, "date_max": df['datetime'].max() if n>0 else None,
        "breakdown": {"ON & Ready": on_ready, "OFF & Ready (Desperdiciado)": off_ready, "OFF & No Ready": off_nready},
        "efficiency_percentage": on_ready / n * 100 if n > 0 else 0,
    }

def create_donut_chart(metrics: Dict[str, Any]) -> go.Figure:
    labels = list(metrics["breakdown"].keys()); values = list(metrics["breakdown"].values())
    colors = [OPTIMITIVE_COLORS['success'], OPTIMITIVE_COLORS['warning'], OPTIMITIVE_COLORS['error']]
    
    # Crear subtítulo con rango de fechas
    date_range_text = ""
    if metrics.get("date_min") and metrics.get("date_max"):
        date_min_str = metrics["date_min"].strftime('%d/%m/%Y')
        date_max_str = metrics["date_max"].strftime('%d/%m/%Y')
        date_range_text = f"<br><span style='font-size:14px; color:{OPTIMITIVE_COLORS['text_secondary']}'>Período: {date_min_str} - {date_max_str}</span><br><br><br><br>"
    
    fig = go.Figure(go.Pie(
        labels=labels, 
        values=values, 
        hole=0.6, 
        sort=False, 
        textinfo="percent+value", 
        textposition="outside",
        marker=dict(colors=colors, line=dict(color="white", width=3)),
        hovertemplate="<b>%{label}</b><br>Registros: %{value:,}<br>Porcentaje: %{percent}<extra></extra>",
        textfont=dict(size=12, color="black"),
        pull=[0.02, 0.02, 0.02]  # Separar ligeramente los segmentos
    ))
    
    # Configurar layout con rango de fechas en el título
    fig.update_layout(
        title=dict(
            text=f"Distribución de Estados del Sistema{date_range_text}", 
            x=0.5,
            y=0.95,  # Posicionar el título más arriba
            font=dict(size=18, color=OPTIMITIVE_COLORS['text_primary'])
        ),
        legend=dict(
            orientation="h", 
            x=0.5, 
            xanchor='center',
            y=-0.1,
            font=dict(size=12)
        ),
        height=600,  # Mitad del tamaño anterior (1200/2)
        margin=dict(l=80, r=80, t=120, b=80),  # Margen superior aumentado para el período
        font=dict(size=12)
    )
    
    return fig

def aggregate_for_plot(df: pl.DataFrame, main_kpi: str, other_kpis: List[str]) -> pl.DataFrame:
    time_range_days = (df['datetime'].max() - df['datetime'].min()).total_seconds() / (3600 * 24)
    if time_range_days > 60: agg_interval = "1h"
    elif time_range_days > 10: agg_interval = "15m"
    elif time_range_days > 2: agg_interval = "5m"
    else: agg_interval = "1m"
    st.caption(f"💡 Los datos del gráfico se agregan en intervalos de **{agg_interval}**.")
    aggs = [
        pl.col(main_kpi).first().alias("open"), pl.col(main_kpi).max().alias("high"),
        pl.col(main_kpi).min().alias("low"), pl.col(main_kpi).last().alias("close"),
        pl.col("OPTIBAT_ON").max().alias("OPTIBAT_ON"), pl.col("OPTIBAT_READY").max().alias("OPTIBAT_READY")
    ]
    for kpi in other_kpis: aggs.append(pl.col(kpi).mean().alias(kpi))
    return df.group_by_dynamic("datetime", every=agg_interval).agg(aggs).sort("datetime")

def create_time_series_chart(df_filtered: pl.DataFrame, selected_vars: Dict[str, str], binary_vars: Dict[str, bool], line_styles: Dict[str, str] = None) -> go.Figure:
    """
    Implementación según el script de referencia:
    - Sombreado por estado (ON, OFF&READY, OFF&NO READY)
    - Flags 0/1 en mismo eje de OPTIBAT_ON con líneas HV
    - KPIs continuos en eje secundario con auto-rango expandido (padding)
    - Zoom rectangular; sin wheel-zoom ni range slider
    - Panorámico y vista por defecto a TODO el período
    """
    if not {"datetime", "OPTIBAT_ON"}.issubset(df_filtered.columns):
        return go.Figure()

    # Inicializar line_styles si no se proporciona
    if line_styles is None:
        line_styles = {}
    
    # Convertir a pandas para facilitar el manejo
    df = df_filtered.to_pandas()
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # === SOMBREADO POR ESTADO ===
    shapes = []
    ready_col = None
    for col in ["OPTIBAT_READY", "Flag_Ready", "Ready"]:
        if col in df.columns:
            ready_col = col
            break
    
    if ready_col:
        # Generar segmentos por estado
        d = df[["datetime", "OPTIBAT_ON", ready_col]].copy()
        d = d.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        d["OPTIBAT_ON"] = pd.to_numeric(d["OPTIBAT_ON"], errors="coerce").fillna(0).astype(int)
        d["ready"] = pd.to_numeric(d[ready_col], errors="coerce").fillna(0).astype(int)
        
        # Calcular status: 0=OFF&NO READY, 1=OFF&READY, 2=ON&READY
        d["status"] = np.select([
            d["OPTIBAT_ON"] == 1,                           # ON (asumimos READY)
            (d["OPTIBAT_ON"] == 0) & (d["ready"] == 1)      # OFF & READY
        ], [2, 1], default=0)  # default: OFF & NO READY
        
        # Detectar cambios de estado
        changes = d["status"].diff().fillna(1).ne(0)
        change_indices = list(d.index[changes]) + [len(d)]
        
        # Crear shapes para sombreado
        for i, start_idx in enumerate(change_indices[:-1]):
            end_idx = change_indices[i + 1] - 1
            if start_idx < len(d) and end_idx < len(d):
                x0 = d.iloc[start_idx]["datetime"]
                x1 = d.iloc[end_idx]["datetime"]
                if x1 == x0:
                    x1 = x1 + pd.Timedelta(minutes=1)
                status = d.iloc[start_idx]["status"]
                
                # Mapeo de colores
                color_map = {
                    2: ColorScheme.GREEN_FILL,    # ON & READY
                    1: ColorScheme.ORANGE_FILL,   # OFF & READY
                    0: ColorScheme.RED_FILL       # OFF & NO READY
                }
                
                shapes.append(dict(
                    type="rect", xref="x", yref="paper",
                    x0=x0, x1=x1, y0=0, y1=1,
                    fillcolor=color_map[status],
                    line=dict(width=0), layer="below"
                ))

    # === ESTADO BASE OPTIBAT_ON ===
    fig.add_trace(go.Scatter(
        x=df["datetime"], 
        y=df["OPTIBAT_ON"], 
        mode="lines",
        line=dict(shape="hv", width=2.4, color=ColorScheme.NEUTRAL_BLUE),
        name="OPTIBAT_ON",
        hovertemplate="<b>%{x}</b><br>Estado: %{y}<extra></extra>"
    ), secondary_y=False)

    # === VARIABLES SELECCIONADAS ===
    cont_series = []  # para calcular rango del eje secundario
    
    for var, color in selected_vars.items():
        if var not in df.columns or var == "OPTIBAT_ON":
            continue
            
        if binary_vars.get(var, False):
            # Variables binarias: mismo eje que OPTIBAT_ON con líneas HV punteadas
            y_vals = pd.to_numeric(df[var], errors="coerce").fillna(0).clip(0, 1)
            line_style = line_styles.get(var, "dot")  # Default "dot" for binary
            fig.add_trace(go.Scatter(
                x=df["datetime"], 
                y=y_vals, 
                mode="lines",
                name=f"{var} (0/1)",
                line=dict(width=1.9, color=color, dash=line_style, shape="hv"),
                hovertemplate=f"<b>%{{x}}</b><br>{var}: %{{y:.0f}}<extra></extra>"
            ), secondary_y=False)
        else:
            # Variables continuas: eje secundario
            y_vals = pd.to_numeric(df[var], errors="coerce")
            cont_series.append(y_vals)
            line_style = line_styles.get(var, "solid")  # Default "solid" for continuous
            fig.add_trace(go.Scatter(
                x=df["datetime"], 
                y=y_vals, 
                mode="lines",
                name=var, 
                line=dict(width=1.9, color=color, dash=line_style),
                hovertemplate=f"<b>%{{x}}</b><br>{var}: %{{y:.3f}}<extra></extra>"
            ), secondary_y=True)

    # === CONFIGURACIÓN DE RANGOS ===
    
    # RANGO X: mostrar SIEMPRE todo el período
    x_min = df["datetime"].min()
    x_max = df["datetime"].max()

    # RANGO Y2 (KPIs continuos): con padding del 10%
    y2_range = None
    if cont_series:
        y_all = pd.concat(cont_series, axis=0).dropna()
        if not y_all.empty:
            y_min, y_max = float(y_all.min()), float(y_all.max())
            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                pad = 0.1 * (y_max - y_min)  # +10% padding
                y2_range = [y_min - pad, y_max + pad]

    # === LAYOUT PANORÁMICO ===
    fig.update_layout(
        shapes=shapes,
        title=dict(text="Serie Temporal OPTIBAT + Variables seleccionadas", x=0.5),
        width=2280,       # MÁS ANCHO (panorámico)
        height=520,       # más bajo (panorámico)
        margin=dict(l=60, r=60, t=70, b=40),
        xaxis=dict(
            title="Fecha y Hora", 
            rangeslider=dict(visible=False),
            range=[x_min, x_max]  # Forzar rango completo
        ),
        yaxis=dict(
            title="Estado / Flags (0=OFF, 1=ON)",
            tickvals=[0, 1], 
            ticktext=["OFF", "ON"], 
            range=[-0.15, 1.15]
        ),
        yaxis2=dict(
            title="KPIs", 
            overlaying="y", 
            side="right",
            range=y2_range if y2_range else None
        ),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
        hovermode="x unified",
        dragmode="zoom",  # zoom rectangular
        uirevision="keep"  # conserva zoom si cambias selección
    )

    # Eliminar herramientas que no sean zoom rectangular
    fig.update_layout(modebar_remove=[
        "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d", "pan2d",
        "select2d", "lasso2d"
    ])

    return fig

# =============== ANÁLISIS DETALLADO DE KPIS ===============

def to01(s):
    """Convierte serie a valores binarios 0/1"""
    if str(s.dtype) == "boolean": 
        return s.astype("Int64").fillna(0).astype(int).clip(0,1)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).clip(0,1)

def ts_on_off_with_means(df: pd.DataFrame, tcol: str, on_col: str, var: str) -> go.Figure:
    """Crea time series continua con medias ON vs OFF - SOLO CUANDO OPTIBAT_READY=1"""
    # FILTRAR SOLO OPTIBAT_READY=1
    ready_col = None
    for col in ["OPTIBAT_READY", "Flag_Ready", "Ready"]:
        if col in df.columns:
            ready_col = col
            break
    
    if ready_col:
        # Filtrar solo registros donde OPTIBAT_READY=1
        df_ready = df[to01(df[ready_col]) == 1].copy()
    else:
        df_ready = df.copy()  # Fallback si no hay columna READY
    
    d = df_ready[[tcol, on_col, var]].copy()
    d[tcol] = pd.to_datetime(d[tcol], errors="coerce")
    d[var] = pd.to_numeric(d[var], errors="coerce")
    d = d.dropna(subset=[tcol]).sort_values(tcol).reset_index(drop=True)
    d['state'] = to01(d[on_col])
    
    fig = go.Figure()
    d['block'] = (d['state'].diff() != 0).cumsum()
    grouped = d.groupby('block')
    legend_added = set()
    
    for name, group in grouped:
        if group.empty: 
            continue
        plot_data = group.copy()
        start_index = group.index[0]
        if start_index > 0:
            plot_data = pd.concat([d.loc[[start_index - 1]], plot_data])
        
        state = group['state'].iloc[0]
        color = "#27AE60" if state == 1 else "#D32F2F"  # Verde para ON, Rojo para OFF
        label = "ON" if state == 1 else "OFF"
        
        fig.add_trace(go.Scatter(
            x=plot_data[tcol], 
            y=plot_data[var], 
            mode='lines', 
            name=label,
            line=dict(color=color, width=2.5), 
            legendgroup=label, 
            showlegend=(label not in legend_added),
            hovertemplate=f'<b>%{{x|%Y-%m-%d %H:%M:%S}}</b><br>{var}: %{{y:.2f}}<br>Estado: {label}<extra></extra>'
        ))
        legend_added.add(label)
    
    # Calcular y agregar líneas de media
    y_valid = d[var].dropna()
    z_valid = d.loc[y_valid.index, 'state']
    mu_off = y_valid[z_valid == 0].mean()
    mu_on = y_valid[z_valid == 1].mean()
    
    xmin, xmax = d[tcol].min(), d[tcol].max()
    if pd.notna(mu_off):
        fig.add_trace(go.Scatter(
            x=[xmin, xmax], 
            y=[mu_off, mu_off], 
            mode="lines", 
            name=f"Media OFF: {mu_off:.2f}", 
            line=dict(color="#D32F2F", dash="dash", width=1.5), 
            hoverinfo="skip"
        ))
    if pd.notna(mu_on):
        fig.add_trace(go.Scatter(
            x=[xmin, xmax], 
            y=[mu_on, mu_on], 
            mode="lines", 
            name=f"Media ON: {mu_on:.2f}", 
            line=dict(color="#27AE60", dash="dash", width=1.5), 
            hoverinfo="skip"
        ))
    
    # Agregar información sobre el filtro READY
    title_text = f"{var} - Serie Temporal Continua (ON vs OFF)"
    if ready_col:
        title_text += f" | Solo {ready_col}=1"
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=18)),
        height=520, 
        margin=dict(l=60, r=30, t=90, b=50), 
        hovermode="closest",
        xaxis=dict(title="Fecha y Hora", rangeslider=dict(visible=True)),
        yaxis=dict(title=var),
        dragmode='zoom'
    )
    
    return fig

def create_boxplot_on_off(df: pd.DataFrame, on_col: str, var: str) -> go.Figure:
    """Crea boxplot comparando ON vs OFF - SOLO CUANDO OPTIBAT_READY=1"""
    # FILTRAR SOLO OPTIBAT_READY=1
    ready_col = None
    for col in ["OPTIBAT_READY", "Flag_Ready", "Ready"]:
        if col in df.columns:
            ready_col = col
            break
    
    if ready_col:
        # Filtrar solo registros donde OPTIBAT_READY=1
        df_ready = df[to01(df[ready_col]) == 1].copy()
    else:
        df_ready = df.copy()  # Fallback si no hay columna READY
    
    z = to01(df_ready[on_col])
    y = pd.to_numeric(df_ready[var], errors="coerce")
    on_data = y[z == 1]
    off_data = y[z == 0]
    
    fig = go.Figure()
    
    # Boxplot para OFF
    fig.add_trace(go.Box(
        y=off_data, 
        name=f"OFF (n={len(off_data):,})", 
        marker_color="#D32F2F", 
        boxmean=True, 
        notched=False, 
        boxpoints='outliers'
    ))
    
    # Boxplot para ON
    fig.add_trace(go.Box(
        y=on_data, 
        name=f"ON (n={len(on_data):,})", 
        marker_color="#27AE60", 
        boxmean=True, 
        notched=False, 
        boxpoints='outliers'
    ))
    
    # Agregar información sobre el filtro READY
    title_text = f"{var} - Boxplot ON vs OFF"
    if ready_col:
        title_text += f" | Solo {ready_col}=1"
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=18)),
        height=650, 
        margin=dict(l=60, r=60, t=70, b=50),
        yaxis=dict(title=var),
        showlegend=True
    )
    
    return fig

def create_distribution_plot(df: pd.DataFrame, on_col: str, var: str) -> go.Figure:
    """Crea gráfico de distribución ON vs OFF con líneas de densidad y medias verticales - SOLO CUANDO OPTIBAT_READY=1"""
    from scipy import stats
    import numpy as np
    
    # FILTRAR SOLO OPTIBAT_READY=1
    ready_col = None
    for col in ["OPTIBAT_READY", "Flag_Ready", "Ready"]:
        if col in df.columns:
            ready_col = col
            break
    
    if ready_col:
        # Filtrar solo registros donde OPTIBAT_READY=1
        df_ready = df[to01(df[ready_col]) == 1].copy()
    else:
        df_ready = df.copy()  # Fallback si no hay columna READY
    
    z = to01(df_ready[on_col])
    y = pd.to_numeric(df_ready[var], errors="coerce")
    on_data = y[z == 1].dropna()
    off_data = y[z == 0].dropna()
    
    fig = go.Figure()
    
    # Calcular medias
    mean_off = off_data.mean() if len(off_data) > 0 else np.nan
    mean_on = on_data.mean() if len(on_data) > 0 else np.nan
    
    # Rango común para ambas distribuciones
    all_data = pd.concat([off_data, on_data])
    x_min, x_max = all_data.min(), all_data.max()
    x_range = np.linspace(x_min, x_max, 300)
    
    # Distribución OFF con líneas suaves - CONTEO DE MUESTRAS
    if len(off_data) > 0:
        try:
            kde_off = stats.gaussian_kde(off_data)
            y_off_density = kde_off(x_range)
            # Convertir densidad a conteo de muestras
            y_off_count = y_off_density * len(off_data) * (x_max - x_min) / 300
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_off_count,
                name=f"OFF (n={len(off_data):,})",
                line=dict(color="red", width=2),
                fill='tonexty' if len(fig.data) == 0 else None,
                fillcolor="rgba(255,99,132,0.3)"
            ))
        except:
            # Fallback si KDE falla - usar histograma con CONTEO
            fig.add_trace(go.Histogram(
                x=off_data,
                name=f"OFF (n={len(off_data):,})",
                opacity=0.6,
                marker_color="rgba(255,99,132,0.6)",
                nbinsx=40,
                histnorm=''  # SIN NORMALIZACIÓN - CONTEO DIRECTO
            ))
    
    # Distribución ON con líneas suaves - CONTEO DE MUESTRAS
    if len(on_data) > 0:
        try:
            kde_on = stats.gaussian_kde(on_data)
            y_on_density = kde_on(x_range)
            # Convertir densidad a conteo de muestras
            y_on_count = y_on_density * len(on_data) * (x_max - x_min) / 300
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_on_count,
                name=f"ON (n={len(on_data):,})",
                line=dict(color="green", width=2),
                fill='tonexty' if len(fig.data) == 0 else None,
                fillcolor="rgba(75,192,192,0.3)"
            ))
        except:
            # Fallback si KDE falla - usar histograma con CONTEO
            fig.add_trace(go.Histogram(
                x=on_data,
                name=f"ON (n={len(on_data):,})",
                opacity=0.6,
                marker_color="rgba(75,192,192,0.6)",
                nbinsx=40,
                histnorm=''  # SIN NORMALIZACIÓN - CONTEO DIRECTO
            ))
    
    # Líneas verticales de media
    if pd.notna(mean_off):
        fig.add_vline(
            x=mean_off, 
            line_dash="dash", 
            line_color="red", 
            line_width=3,
            annotation_text=f"μ OFF: {mean_off:.2f}",
            annotation_position="top",
            annotation=dict(
                font=dict(color="red", size=12),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            )
        )
    
    if pd.notna(mean_on):
        fig.add_vline(
            x=mean_on, 
            line_dash="dash", 
            line_color="green", 
            line_width=3,
            annotation_text=f"μ ON: {mean_on:.2f}",
            annotation_position="top",
            annotation=dict(
                font=dict(color="green", size=12),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="green",
                borderwidth=1
            )
        )
    
    # Agregar información sobre el filtro READY
    title_text = f"{var} · Distribución"
    if ready_col:
        title_text += f" | Solo {ready_col}=1"
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=18)),
        height=500,
        margin=dict(l=60, r=30, t=100, b=50),
        xaxis=dict(title=var),
        yaxis=dict(title="Cantidad de muestras"),  # CAMBIADO DE DENSIDAD A CANTIDAD
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
    )
    
    return fig

def create_statistics_table_detailed(df: pd.DataFrame, on_col: str, var: str) -> go.Figure:
    """Crea tabla de estadísticas detalladas ON vs OFF - SOLO CUANDO OPTIBAT_READY=1"""
    # FILTRAR SOLO OPTIBAT_READY=1
    ready_col = None
    for col in ["OPTIBAT_READY", "Flag_Ready", "Ready"]:
        if col in df.columns:
            ready_col = col
            break
    
    if ready_col:
        # Filtrar solo registros donde OPTIBAT_READY=1
        df_ready = df[to01(df[ready_col]) == 1].copy()
    else:
        df_ready = df.copy()  # Fallback si no hay columna READY
    
    z = to01(df_ready[on_col])
    y = pd.to_numeric(df_ready[var], errors="coerce")
    
    def _stats(s):
        s_num = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
        if s_num.empty: 
            return [len(s), 0] + [np.nan] * 7
        q = s_num.quantile([0.25, 0.5, 0.75])
        return [
            len(s), 
            len(s_num), 
            s_num.mean(), 
            s_num.std(ddof=1), 
            s_num.min(), 
            q.loc[0.25], 
            q.loc[0.5], 
            q.loc[0.75], 
            s_num.max()
        ]
    
    off_data = y[z == 0]
    on_data = y[z == 1]
    
    st_off = _stats(off_data)
    st_on = _stats(on_data)
    
    # Calcular delta
    delta = (st_on[2] - st_off[2]) if pd.notna(st_on[2]) and pd.notna(st_off[2]) else np.nan
    pct = (delta / st_off[2] * 100) if pd.notna(delta) and pd.notna(st_off[2]) and st_off[2] != 0 else np.nan
    delta_txt = f"{delta:+.2f} ({pct:+.2f}%)" if pd.notna(delta) and pd.notna(pct) else "N/A"
    
    # Formatear datos
    f = lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
    
    metrics = [
        "Count (Total/Válidos)", 
        "Mean", 
        "Std Dev", 
        "Min", 
        "Q1 (25%)", 
        "Median", 
        "Q3 (75%)", 
        "Max", 
        "Δ Mean (ON−OFF)"
    ]
    
    col_off = [f"{st_off[0]:,} ({st_off[1]:,} v.)"] + [f(x) for x in st_off[2:]] + [delta_txt]
    col_on = [f"{st_on[0]:,} ({st_on[1]:,} v.)"] + [f(x) for x in st_on[2:]] + [""]
    
    # Colores para filas alternadas
    row_colors = ["#ffffff" if i % 2 == 0 else "#f8f9fa" for i in range(len(metrics))]
    
    # Colores para columnas OFF y ON
    col_off_colors = row_colors.copy()
    col_on_colors = row_colors.copy()
    
    # Color condicional para Δ Mean (última fila)
    if pd.notna(delta):
        delta_color = "#28a745" if delta > 0 else "#0066cc"  # Verde si positivo, azul si negativo
        col_off_colors[-1] = delta_color
    
    # Texto en blanco para el delta con color de fondo
    font_colors_off = ["#1f2937"] * (len(metrics) - 1) + ["white" if pd.notna(delta) else "#1f2937"]
    font_colors_on = ["#1f2937"] * len(metrics)
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Métrica</b>", "<b>OFF</b>", "<b>ON</b>"],
            fill_color="#1f2937",
            font=dict(color="white", size=16),
            align="left",
            height=40
        ),
        cells=dict(
            values=[metrics, col_off, col_on],
            fill_color=[row_colors, col_off_colors, col_on_colors],
            font=dict(
                color=[["#1f2937"] * len(metrics), font_colors_off, font_colors_on],
                size=14
            ),
            align="left",
            height=36
        )
    )])
    
    # Agregar información sobre el filtro READY
    title_text = f"{var} - Estadísticas Detalladas"
    if ready_col:
        title_text += f" | Solo {ready_col}=1"
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=18)),
        height=460,
        margin=dict(l=14, r=14, t=50, b=10)
    )
    
    return fig

def create_statistics_table(df: pl.DataFrame, kpis: List[str]) -> None:
    """Genera tabla de estadísticas descriptivas para los KPIs analizados."""
    if not kpis:
        st.info("Selecciona KPIs en la sección temporal para ver estadísticas.")
        return
    
    try:
        stats_df = df.select(kpis).describe()
        st.dataframe(stats_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error calculando estadísticas: {e}")

# =============== CSS PROFESIONAL (Del script Jupyter) ===============
PROFESSIONAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', sans-serif; background: #ffffff; color: #1e293b; line-height: 1.6; }
.container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
.report-header { background: linear-gradient(135deg, #2F80ED 0%, #2563eb 100%); border-radius: 1rem; padding: 2.5rem; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.1); color: white; position: relative; overflow: hidden; }
.report-header::before { content: ''; position: absolute; top: -50%; right: -10%; width: 50%; height: 200%; background: rgba(255,255,255,0.1); transform: rotate(35deg); }
.h1 { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; color: white; }
.card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 1.5rem; margin: 1.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.07); }
.kpi-block { margin: 2rem 0; padding: 2rem; border-radius: 1rem; background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
.kpi-title { font-weight: 700; font-size: 1.5rem; margin-bottom: 1rem; color: #0B1220; display: flex; align-items: center; gap: 0.5rem; }
.kpi-title::before { content: '📊'; font-size: 1.25rem; }
.alert { padding: 1rem 1.5rem; border-radius: 0.75rem; margin: 1.5rem 0; border-left: 4px solid; }
.alert-danger { background: #fee2e2; border-color: #D32F2F; color: #991b1b; }
.alert-info { background: #e0f2fe; border-color: #3b82f6; color: #1e40af; }
.alert-warning { background: #fef9c3; border-color: #f59e0b; color: #92400e; }
.metadata-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1.5rem; }
.metadata-item { background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 0.5rem; }
.metadata-label { font-size: 0.875rem; margin-bottom: 0.25rem; color: #1e293b; font-weight: 500; }
.metadata-value { font-size: 1.125rem; font-weight: 700; color: #0B1220; }
.date-filter-summary { background: #f8fafc; border: 2px solid #2F80ED; border-radius: 0.75rem; padding: 1rem; margin: 1rem 0; text-align: center; }
.date-range-highlight { color: #2F80ED; font-weight: 700; font-size: 1.1rem; }
hr { border: 0; border-top: 2px solid #e2e8f0; margin: 2rem 0; }
.section { margin: 3rem 0; }
.section-title { font-size: 1.875rem; font-weight: 700; color: #0B1220; margin-bottom: 1.5rem; padding-bottom: 0.75rem; border-bottom: 3px solid #2F80ED; }
.report-footer { margin-top: 4rem; padding: 2rem; background: #f8fafc; border-radius: 1rem; text-align: center; color: #64748b; border: 1px solid #e2e8f0; }
.small { font-size: 0.875rem; opacity: 0.8; color: #64748b; }
.filter-preset { background: #e0f2fe; border: 1px solid #2F80ED; border-radius: 0.5rem; padding: 0.5rem; margin: 0.25rem; display: inline-block; cursor: pointer; font-size: 0.9rem; color: #2F80ED; font-weight: 500; }
.filter-preset:hover { background: #2F80ED; color: white; }
</style>
"""

# =============== HTML REPORT GENERATION ===============
def fig_to_html_div(fig):
    """Convierte figura Plotly a HTML div con configuración personalizada"""
    config = {
        'displaylogo': False, 
        'modeBarButtonsToAdd': ['drawrect', 'eraseshape'], 
        'modeBarButtonsToRemove': ['lasso2d']
    }
    return fig.to_html(include_plotlyjs="cdn", full_html=False, config=config)

def build_html(title, sections):
    """Construye HTML completo con título y secciones"""
    return f"""<!DOCTYPE html>
<html lang='es'>
<head>
    <meta charset='utf-8'>
    <title>{title}</title>
    {PROFESSIONAL_CSS}
</head>
<body>
    <div class='container'>
        {''.join(sections)}
    </div>
</body>
</html>"""

def period_label(df: pd.DataFrame, tcol: str):
    """Genera etiqueta de período para el reporte"""
    if tcol and tcol in df.columns and not df.empty:
        d = pd.to_datetime(df[tcol], errors="coerce").dropna()
        if not d.empty:
            days = (d.max().normalize() - d.min().normalize()).days + 1
            return f"{d.min():%Y-%m-%d} → {d.max():%Y-%m-%d} · {days:,} días"
    return f"{len(df):,} registros"

def create_html_report(df_analysis: pd.DataFrame, selected_kpis: list, on_col: str, 
                      time_col: str, report_title: str, dataset_name: str,
                      start_date=None, end_date=None) -> str:
    """
    Genera reporte HTML completo con todos los análisis KPI.
    Sigue la estrategia del script Jupyter con CSS profesional.
    Incluye información sobre filtros de fecha aplicados.
    """
    sections = []
    
    # Detectar columna READY para información
    ready_col = None
    for col in ["OPTIBAT_READY", "Flag_Ready", "Ready"]:
        if col in df_analysis.columns:
            ready_col = col
            break
    
    # Header con información de análisis
    filter_summary = ""
    
    # Agregar información de filtro de fechas
    if start_date and end_date:
        filter_summary += f"""
        <div class='date-filter-summary'>
            <h3 style='margin: 0; color: #2F80ED;'>📅 Filtro de Fechas Aplicado</h3>
            <div class='date-range-highlight'>{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}</div>
            <div style='margin-top: 0.5rem; font-size: 0.9rem; color: #64748b;'>
                Período: {(end_date - start_date).days + 1} días | 
                Registros filtrados: {len(df_analysis):,}
            </div>
        </div>
        """
    
    if ready_col:
        ready_count = (to01(df_analysis[ready_col]) == 1).sum()
        filter_summary += f"""
        <div class='date-filter-summary' style='margin-top: 1rem;'>
            <h3 style='margin: 0; color: #2F80ED;'>🔍 Filtrado OPTIBAT_READY=1 Aplicado</h3>
            <div class='date-range-highlight'>Analizando {ready_count:,} registros operativos de {len(df_analysis):,} totales</div>
            <div style='margin-top: 0.5rem; font-size: 0.9rem; color: #64748b;'>
                Solo se incluyen datos cuando el sistema está completamente operativo
            </div>
        </div>
        """
    
    sections.append(f"""
    <div class='report-header'>
        <div class='h1'>{report_title}</div>
        {filter_summary}
        <div class='metadata-grid'>
            <div class='metadata-item'>
                <div class='metadata-label'>Generación</div>
                <div class='metadata-value'>{datetime.now():%Y-%m-%d %H:%M}</div>
            </div>
            <div class='metadata-item'>
                <div class='metadata-label'>Dataset</div>
                <div class='metadata-value'>{dataset_name}</div>
            </div>
            <div class='metadata-item'>
                <div class='metadata-label'>KPIs Analizados</div>
                <div class='metadata-value'>{len(selected_kpis)}</div>
            </div>
            <div class='metadata-item'>
                <div class='metadata-label'>Período</div>
                <div class='metadata-value'>{period_label(df_analysis, time_col)}</div>
            </div>
        </div>
    </div>
    """)
    
    # Análisis de KPIs
    if selected_kpis:
        sections.append(f"""
        <div class='section'>
            <div class='section-title'>📈 Análisis Detallado de KPIs</div>
        </div>
        """)
        
        # Nota sobre filtrado
        if ready_col:
            sections.append(f"""
            <div class='alert alert-info'>
                ℹ️ <b>Nota:</b> Todos los análisis se realizan únicamente sobre registros donde 
                <code>{ready_col} = 1</code>, garantizando que solo se analicen datos del sistema 
                completamente operativo.
            </div>
            """)
        
        # Análisis por cada KPI
        for i, var in enumerate(selected_kpis, 1):
            try:
                sections.append(f"""
                <div class='kpi-block'>
                    <div class='kpi-title'>{var}</div>
                """)
                
                # Generar los 4 gráficos para cada KPI
                charts_html = generate_html_charts(df_analysis, on_col, time_col, var)
                sections.append(charts_html)
                
                sections.append("</div>")
                
            except Exception as e:
                sections.append(f"""
                <div class='alert alert-danger'>
                    ❌ Error en KPI <strong>{var}</strong>: {str(e)}
                </div>
                """)
    
    # Footer
    sections.append(f"""
    <div class='report-footer'>
        <h3 style='color: #2F80ED; margin-bottom: 1rem;'>OPTIBAT Analytics - Reporte Completo</h3>
        <div><strong>Generación:</strong> {datetime.now():%Y-%m-%d %H:%M:%S}</div>
        <div><strong>Dataset:</strong> {dataset_name}</div>
        <div><strong>Variables Analizadas:</strong> {len(selected_kpis)}</div>
        <div class='small' style='margin-top: 0.5rem;'>© 2025 Desarrollado por Juan Cruz E. - OPTIMITIVE</div>
    </div>
    """)
    
    # Construir HTML final
    return build_html(report_title, sections)

def generate_html_charts(df: pd.DataFrame, on_col: str, time_col: str, kpi_var: str) -> str:
    """
    Genera los 4 gráficos HTML para un KPI específico.
    Retorna el HTML con todos los gráficos embebidos.
    """
    charts_html = []
    
    try:
        # 1. Time Series Continua
        ts_fig = ts_on_off_with_means(df, time_col, on_col, kpi_var)
        charts_html.append(f"""
        <h3 style='margin-top: 2rem;'>📈 Serie Temporal Continua (ON vs OFF)</h3>
        {fig_to_html_div(ts_fig)}
        <hr>
        """)
    except Exception as e:
        charts_html.append(f"<div class='alert alert-danger'>Error en serie temporal: {e}</div>")
    
    try:
        # 2. Boxplot
        box_fig = create_boxplot_on_off(df, on_col, kpi_var)
        charts_html.append(f"""
        <h3>📊 Boxplot ON vs OFF</h3>
        {fig_to_html_div(box_fig)}
        <hr>
        """)
    except Exception as e:
        charts_html.append(f"<div class='alert alert-danger'>Error en boxplot: {e}</div>")
    
    try:
        # 3. Distribución
        dist_fig = create_distribution_plot(df, on_col, kpi_var)
        charts_html.append(f"""
        <h3>📈 Distribución</h3>
        {fig_to_html_div(dist_fig)}
        <hr>
        """)
    except Exception as e:
        charts_html.append(f"<div class='alert alert-danger'>Error en distribución: {e}</div>")
    
    try:
        # 4. Estadísticas
        stats_fig = create_statistics_table_detailed(df, on_col, kpi_var)
        charts_html.append(f"""
        <h3>📋 Estadísticas Detalladas</h3>
        {fig_to_html_div(stats_fig)}
        """)
    except Exception as e:
        charts_html.append(f"<div class='alert alert-danger'>Error en estadísticas: {e}</div>")
    
    return ''.join(charts_html)

def render_dataframe_filter_ui():
    """Renderiza la interfaz de filtrado de DataFrames."""
    st.header("🔍 Filtrado Avanzado de DataFrames")
    
    # Obtener DataFrames disponibles
    available_dfs = get_available_dataframes()
    
    if not available_dfs:
        st.warning("No hay DataFrames disponibles para filtrar. Primero carga datos.")
        return
    
    # Selector de DataFrame
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_df_name = st.selectbox(
            "Seleccionar DataFrame:",
            options=list(available_dfs.keys()),
            key="df_filter_selector",
            help="Elige el DataFrame que quieres filtrar"
        )
    
    with col2:
        st.metric("Filas", f"{len(available_dfs[selected_df_name]):,}")
        st.metric("Columnas", len(available_dfs[selected_df_name].columns))
    
    current_df = available_dfs[selected_df_name]
    
    # Información del DataFrame seleccionado
    st.subheader(f"📊 DataFrame: {selected_df_name}")
    
    with st.expander("👁️ Vista previa de columnas"):
        cols_preview = pd.DataFrame({"Columnas": list(current_df.columns)})
        st.dataframe(cols_preview, height=200)
    
    # Selector de columnas para filtros de rango
    st.subheader("⚙️ Selección de Variables para Filtros de Rango")
    
    # Obtener columnas numéricas
    numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
    
    selected_numeric_cols = st.multiselect(
        "Selecciona columnas numéricas para crear filtros de rango:",
        options=numeric_cols,
        help="Solo se mostrarán sliders para las columnas seleccionadas",
        key="numeric_cols_selector"
    )
    
    # Generar sliders de rango para columnas seleccionadas
    range_filters = {}
    if selected_numeric_cols:
        st.subheader("📏 Filtros de Rango (min ≤ valor ≤ max)")
        
        for col in selected_numeric_cols:
            col_data = pd.to_numeric(current_df[col], errors='coerce').dropna()
            if not col_data.empty:
                min_val, max_val = float(col_data.min()), float(col_data.max())
                if min_val < max_val:
                    step = (max_val - min_val) / 100.0
                    
                    # Layout con inputs numéricos y slider
                    st.write(f"**{col}**")
                    
                    # Crear columnas para inputs y slider
                    col_min, col_slider, col_max = st.columns([1, 3, 1])
                    
                    # Valores por defecto del slider
                    slider_default = (min_val, max_val)
                    if f"range_{col}" in st.session_state:
                        slider_default = st.session_state[f"range_{col}"]
                    
                    # Input numérico para valor mínimo (sin restricciones de rango)
                    with col_min:
                        min_input_raw = st.number_input(
                            "Min:",
                            value=slider_default[0],
                            step=step,
                            key=f"min_input_{col}",
                            help=f"Valor mínimo para {col} (se auto-ajustará si está fuera de rango)"
                        )
                        # Auto-ajustar si está fuera de rango
                        min_input = max(min_val, min(min_input_raw, max_val))
                    
                    # Slider principal
                    with col_slider:
                        # Asegurar que los valores del slider sean válidos
                        current_min = max(min_val, min_input)
                        current_max = min(max_val, st.session_state.get(f"max_input_{col}", slider_default[1]))
                        
                        range_val = st.slider(
                            f"Rango de {col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=(current_min, current_max),
                            step=step,
                            key=f"range_{col}",
                            help=f"Arrastra para filtrar {col}",
                            label_visibility="collapsed"
                        )
                    
                    # Input numérico para valor máximo (sin restricciones de rango)
                    with col_max:
                        max_input_raw = st.number_input(
                            "Max:",
                            value=range_val[1],
                            step=step,
                            key=f"max_input_{col}",
                            help=f"Valor máximo para {col} (se auto-ajustará si está fuera de rango)"
                        )
                        # Auto-ajustar si está fuera de rango
                        max_input = min(max_val, max(max_input_raw, min_val))
                    
                    # Sincronización: usar los valores de los inputs si son diferentes del slider
                    final_min = min_input
                    final_max = max_input
                    
                    # Asegurar que min <= max
                    if final_min > final_max:
                        final_min = final_max
                    
                    # Solo agregar al filtro si es diferente del rango completo
                    if (final_min, final_max) != (min_val, max_val):
                        range_filters[col] = (final_min, final_max)
                    
                    # Mostrar valores actuales con indicación de auto-ajuste
                    caption_text = f"Filtro activo: {final_min:.3f} ≤ {col} ≤ {final_max:.3f}"
                    
                    # Indicar si se auto-ajustaron valores
                    adjustments = []
                    if min_input_raw < min_val:
                        adjustments.append(f"Min ajustado: {min_input_raw:.3f} → {final_min:.3f}")
                    elif min_input_raw > max_val:
                        adjustments.append(f"Min ajustado: {min_input_raw:.3f} → {final_min:.3f}")
                    
                    if max_input_raw > max_val:
                        adjustments.append(f"Max ajustado: {max_input_raw:.3f} → {final_max:.3f}")
                    elif max_input_raw < min_val:
                        adjustments.append(f"Max ajustado: {max_input_raw:.3f} → {final_max:.3f}")
                    
                    if adjustments:
                        st.caption(caption_text)
                        st.info("⚙️ Auto-ajustado: " + " | ".join(adjustments))
                    else:
                        st.caption(caption_text)
                    
                    st.markdown("---")
    
    # Filtro de búsqueda de texto
    st.subheader("🔎 Búsqueda de Texto")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_text = st.text_input(
            "Texto a buscar:",
            help="Busca este texto en las columnas seleccionadas",
            key="search_text_input"
        )
    
    with col2:
        use_regex = st.checkbox("Usar regex", key="use_regex_checkbox")
    
    with col3:
        case_sensitive = st.checkbox("Sensible a mayúsculas", key="case_sensitive_checkbox")
    
    # Selector de columnas para búsqueda
    search_scope = st.selectbox(
        "Buscar en:",
        options=["Todas las columnas", "Columnas seleccionadas", "Elegir columnas específicas"],
        key="search_scope_selector"
    )
    
    search_cols = []
    if search_scope == "Todas las columnas":
        search_cols = list(current_df.columns)
    elif search_scope == "Columnas seleccionadas":
        search_cols = selected_numeric_cols if selected_numeric_cols else list(current_df.columns)
    else:
        search_cols = st.multiselect(
            "Seleccionar columnas para búsqueda:",
            options=list(current_df.columns),
            key="search_specific_cols"
        )
    
    # Botón único de acción
    st.subheader("🎯 Filtrar, Exportar y Guardar")
    
    # UN SOLO BOTÓN QUE HACE TODO
    process_all = st.button(
        "🚀 FILTRAR, EXPORTAR Y GUARDAR TODO",
        help="Un solo clic: Aplicar filtros + Guardar CSV en Descargas + Guardar en memoria automáticamente",
        use_container_width=True,
        type="primary"
    )
    
    # Aplicar filtros
    if process_all:
        # Aplicar filtros de rango
        filtered_df = apply_range_filters(current_df, range_filters)
        
        # Aplicar búsqueda de texto
        if search_text and search_cols:
            filtered_df = apply_text_search(
                filtered_df, search_text, search_cols, use_regex, case_sensitive
            )
        
        # Mostrar resultado
        st.subheader("📋 Resultado del Filtrado")
        
        reduction_pct = (1 - len(filtered_df) / len(current_df)) * 100 if len(current_df) > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filas originales", f"{len(current_df):,}")
        with col2:
            st.metric("Filas filtradas", f"{len(filtered_df):,}")
        with col3:
            st.metric("Reducción", f"{reduction_pct:.1f}%")
        
        if len(filtered_df) > 0:
            # Mostrar preview del resultado
            st.write("**Vista previa del resultado filtrado:**")
            preview_rows = min(1000, len(filtered_df))
            st.dataframe(filtered_df.head(preview_rows))
            
            if len(filtered_df) > preview_rows:
                st.info(f"Mostrando las primeras {preview_rows:,} filas de {len(filtered_df):,} total.")
            
            # Guardar en session state
            st.session_state.df_filtrado = filtered_df
            
            # ===== WORKFLOW COMPLETO AUTOMATIZADO =====
            # EJECUTA TODO AUTOMÁTICAMENTE CON UN SOLO BOTÓN
            
            # 1. FILTRAR (ya hecho arriba)
            st.info("🔍 **PASO 1/4**: Filtros aplicados correctamente")
            
            # 2. LIMPIAR Y PREPARAR PARA EXPORTACIÓN
            df_clean = clean_dataframe_for_export(filtered_df)
            st.info("🧹 **PASO 2/4**: Datos limpiados para exportación")
            
            # 3. EXPORTAR Y GUARDAR DIRECTAMENTE EN DESCARGAS
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{selected_df_name}_filtrado_{timestamp}.csv"
            
            # Ruta directa a carpeta Descargas
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
            full_path = os.path.join(downloads_path, filename)
            
            try:
                # Guardar como CSV con valores nulos como 'Null' (criterio del script de referencia)
                df_clean.to_csv(full_path, index=False, na_rep='Null')
                st.success(f"✅ **PASO 3/4**: Archivo CSV guardado directamente en Descargas")
                st.info(f"📁 **Ubicación**: {full_path}")
                
                # También ofrecer descarga por navegador como backup
                csv_data = df_clean.to_csv(index=False, na_rep='Null').encode('utf-8')
                st.download_button(
                    label="🔄 Descarga Manual (Backup)",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True,
                    help="Solo usar si el guardado automático falló"
                )
                
            except Exception as e:
                st.error(f"❌ Error al guardar automáticamente: {str(e)}")
                # Fallback: solo descarga por navegador
                csv_data = df_clean.to_csv(index=False, na_rep='Null').encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar CSV (Manual)",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )
                st.info("💾 **PASO 3/4**: Descarga manual disponible")
            
            # 4. GUARDAR AUTOMÁTICAMENTE EN MEMORIA
            new_df_name_auto = f"{selected_df_name}_exportado_{timestamp}"
            st.session_state.available_dataframes[new_df_name_auto] = filtered_df.copy()
            
            st.success("✅ **PASO 4/4**: DataFrame guardado automáticamente en memoria")
            st.success(f"🎯 **WORKFLOW COMPLETADO**: {filename}")
            st.info(f"📊 DataFrame guardado como: **{new_df_name_auto}** (disponible en selectores)")
            
            # Mostrar resumen del workflow
            with st.expander("📋 Resumen del Workflow Ejecutado"):
                st.write("**✅ Operaciones completadas automáticamente:**")
                st.write("1. 🔍 **Filtrado**: Aplicados filtros de rango y texto")
                st.write("2. 🧹 **Limpieza**: Datos preparados para exportación")  
                st.write("3. 💾 **Exportación**: CSV guardado directamente en Descargas")
                st.write("4. 💿 **Memoria**: DataFrame guardado automáticamente")
                st.write(f"**Archivo generado**: `{filename}`")
                st.write(f"**DataFrame en memoria**: `{new_df_name_auto}`")
                st.write(f"**Registros procesados**: {len(filtered_df):,}")
            
            # Auto-refresh para mostrar el nuevo DF en selectores
            st.rerun()
        
        else:
            st.warning("⚠️ No se encontraron filas que cumplan los criterios de filtrado.")
    
    return

def is_binary_variable(df: pl.DataFrame, col: str) -> bool:
    """Detecta si una variable es binaria (flag 0/1) usando la lógica correcta."""
    try:
        # Convertir a pandas Series para usar la misma lógica del script de referencia
        series = df.select(pl.col(col)).to_pandas()[col]
        s = pd.to_numeric(series, errors="coerce").dropna()
        
        if s.empty:
            return False
        
        # Obtener valores únicos
        unique_vals = np.unique(s.values)
        
        # Es binaria si tiene máximo 3 valores únicos y todos están en el conjunto {0, 1}
        return len(unique_vals) <= 3 and set(np.round(unique_vals).astype(int)).issubset({0, 1})
    except:
        return False

def create_variable_selector(df_filtered: pl.DataFrame) -> tuple:
    """Crea el selector de variables con lista desplegable y buscador."""
    # Obtener todas las columnas numéricas excluyendo datetime
    all_numeric_cols = [c for c in df_filtered.columns 
                       if df_filtered[c].dtype in [pl.Float64, pl.Int32, pl.Int64] 
                       and c not in ["OPTIBAT_ON", "OPTIBAT_READY"]]
    
    # Detectar variables binarias
    binary_vars = {}
    for col in all_numeric_cols:
        binary_vars[col] = is_binary_variable(df_filtered, col)
    
    # Preparar opciones con iconos discretos
    variable_options = []
    for col in all_numeric_cols:
        icon = "●" if binary_vars.get(col, False) else "■"  # Símbolos más profesionales
        category = "Control" if any(keyword in col.lower() for keyword in ['sp', 'setpoint', 'target', 'opt_']) else "Proceso"
        variable_options.append(f"{icon} {col} ({category})")
    
    # Lista desplegable múltiple con buscador integrado
    st.write("**Selección de Variables para Análisis**")
    
    selected_variable_labels = st.multiselect(
        "Variables para Time Series:",
        options=variable_options,
        default=[],
        placeholder="Busca y selecciona variables...",
        help="Busca por nombre de variable. ● = Binaria (0/1), ■ = Continua",
        key="variable_multiselect"
    )
    
    # Procesar selecciones con configuración individual de color y estilo
    selected_vars = {}
    line_styles = {}
    
    if selected_variable_labels:
        st.write("**Configuración Individual de Variables**")
        
        for i, label in enumerate(selected_variable_labels):
            # Extraer nombre de variable del label
            var_name = label.split(' (')[0][2:]  # Remover icono y categoría
            is_binary = binary_vars.get(var_name, False)
            
            # Crear columnas para configuración individual
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"{'●' if is_binary else '■'} **{var_name}**")
            
            with col2:
                # Selector de color individual
                default_color = ColorScheme.PALETTE[i % len(ColorScheme.PALETTE)]
                color_key = f"color_{var_name}_{i}"
                selected_color = st.color_picker(
                    f"Color", 
                    value=default_color, 
                    key=color_key,
                    help=f"Color para {var_name}"
                )
                selected_vars[var_name] = selected_color
            
            with col3:
                # Selector de estilo de línea para todas las variables
                style_key = f"style_{var_name}_{i}"
                default_style = "dot" if is_binary else "solid"
                line_style = st.selectbox(
                    "Trazo",
                    options=["solid", "dash", "dot", "dashdot"],
                    index=["solid", "dash", "dot", "dashdot"].index(default_style),
                    key=style_key,
                    help=f"Tipo de trazo para {var_name} ({'Binaria' if is_binary else 'Continua'})"
                )
                line_styles[var_name] = line_style
        
        st.success(f"✓ {len(selected_vars)} variables configuradas")
    else:
        st.info("Selecciona variables de la lista desplegable para personalizar colores y estilos")
    
    return selected_vars, binary_vars, line_styles

# ==============================================================================
# FUNCIONES DE EXPORTACIÓN HTML
# ==============================================================================

def create_html_report(
    df_analysis: pd.DataFrame, 
    selected_kpis: list, 
    on_col: str, 
    time_col: str,
    report_title: str,
    dataset_name: str = "DataFrame Principal",
    start_date=None,
    end_date=None,
    df_efficiency: pd.DataFrame = None
) -> str:
    """
    Genera un reporte HTML completo con todos los gráficos de análisis KPI
    df_analysis: DataFrame para análisis detallado de KPIs
    df_efficiency: DataFrame para análisis de eficiencia general (si no se proporciona, usa df_analysis)
    """
    
    # Funciones auxiliares que necesitamos para el HTML
    def to01(s):
        """Convierte series a binario 0/1"""
        return pd.to_numeric(s, errors='coerce').fillna(0).clip(0, 1).astype(int)
    
    def generate_html_charts(df, on_col, time_col, kpi_var):
        """Genera todos los gráficos HTML para un KPI específico"""
        charts_html = []
        
        # FILTRAR SOLO OPTIBAT_READY=1
        ready_col = None
        for col in ["OPTIBAT_READY", "Flag_Ready", "Ready"]:
            if col in df.columns:
                ready_col = col
                break
        
        if ready_col:
            df_ready = df[to01(df[ready_col]) == 1].copy()
            ready_info = f" | Solo OPTIBAT_READY=1 ({len(df_ready):,} registros)"
        else:
            df_ready = df.copy()
            ready_info = f" | Todos los registros ({len(df_ready):,})"
        
        # 1. SERIE TEMPORAL
        try:
            fig1 = go.Figure()
            
            # Datos ON vs OFF
            df_ready['ON_STATE'] = to01(df_ready[on_col])
            on_data = df_ready[df_ready['ON_STATE'] == 1]
            off_data = df_ready[df_ready['ON_STATE'] == 0]
            
            # Líneas ON/OFF
            if len(off_data) > 0:
                fig1.add_trace(go.Scatter(
                    x=off_data[time_col], y=off_data[kpi_var],
                    mode='lines+markers', name='OFF', 
                    line=dict(color='red', width=2),
                    marker=dict(size=4)
                ))
            
            if len(on_data) > 0:
                fig1.add_trace(go.Scatter(
                    x=on_data[time_col], y=on_data[kpi_var],
                    mode='lines+markers', name='ON',
                    line=dict(color='green', width=2),
                    marker=dict(size=4)
                ))
            
            # Líneas de media
            if len(off_data) > 0:
                mean_off = off_data[kpi_var].mean()
                fig1.add_hline(y=mean_off, line_dash="dash", line_color="red", 
                              annotation_text=f"Media OFF: {mean_off:.2f}")
            if len(on_data) > 0:
                mean_on = on_data[kpi_var].mean()
                fig1.add_hline(y=mean_on, line_dash="dash", line_color="green",
                              annotation_text=f"Media ON: {mean_on:.2f}")
            
            fig1.update_layout(
                title=f"{kpi_var} - Serie Temporal Continua (ON vs OFF){ready_info}",
                xaxis_title="Tiempo",
                yaxis_title=kpi_var,
                template="plotly_white",
                height=500
            )
            charts_html.append(pio.to_html(fig1, include_plotlyjs='inline', div_id=f"chart1_{kpi_var}"))
        except Exception as e:
            charts_html.append(f"<div class='error'>Error en serie temporal para {kpi_var}: {str(e)}</div>")
        
        # 2. BOXPLOT
        try:
            fig2 = go.Figure()
            
            if len(off_data) > 0:
                fig2.add_trace(go.Box(
                    y=off_data[kpi_var], name='OFF',
                    boxpoints='outliers', marker_color='red'
                ))
            if len(on_data) > 0:
                fig2.add_trace(go.Box(
                    y=on_data[kpi_var], name='ON',
                    boxpoints='outliers', marker_color='green'
                ))
            
            fig2.update_layout(
                title=f"{kpi_var} - Boxplot ON vs OFF{ready_info}",
                yaxis_title=kpi_var,
                template="plotly_white",
                height=650
            )
            charts_html.append(pio.to_html(fig2, include_plotlyjs=False, div_id=f"chart2_{kpi_var}"))
        except Exception as e:
            charts_html.append(f"<div class='error'>Error en boxplot para {kpi_var}: {str(e)}</div>")
        
        # 3. DISTRIBUCIÓN
        try:
            # Usar la función create_distribution_plot que incluye las líneas de promedio
            dist_fig = create_distribution_plot(df_ready, on_col, kpi_var)
            charts_html.append(pio.to_html(dist_fig, include_plotlyjs=False, div_id=f"chart3_{kpi_var}"))
        except Exception as e:
            charts_html.append(f"<div class='error'>Error en distribución para {kpi_var}: {str(e)}</div>")
        
        # 4. TABLA DE ESTADÍSTICAS - FORMATO PROFESIONAL
        try:
            # Usar la función create_statistics_table_detailed para generar la tabla
            stats_fig = create_statistics_table_detailed(df_ready, on_col, kpi_var)
            charts_html.append(pio.to_html(stats_fig, include_plotlyjs=False, div_id=f"chart4_{kpi_var}"))
        except Exception as e:
            charts_html.append(f"<div class='error'>Error en estadísticas para {kpi_var}: {str(e)}</div>")
            
        return charts_html
    
    # GENERAR HTML COMPLETO
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{report_title}</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                color: #2c3e50;
            }}
            .header {{
                background: linear-gradient(135deg, #E31E32 0%, #CC1A2C 100%);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                margin-bottom: 2rem;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5rem;
                font-weight: 900;
            }}
            .header p {{
                margin: 0.5rem 0 0 0;
                font-size: 1.2rem;
            }}
            .kpi-section {{
                background: white;
                margin-bottom: 3rem;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            .kpi-title {{
                color: #E31E32;
                border-bottom: 3px solid #E31E32;
                padding-bottom: 0.5rem;
                margin-bottom: 2rem;
            }}
            .chart-container {{
                margin-bottom: 2rem;
                padding: 1rem;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                background: white;
            }}
            .info-box {{
                background: #e9ecef;
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 2rem;
                border-left: 4px solid #E31E32;
            }}
            .error {{
                color: #dc3545;
                background: #f8d7da;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }}
            .footer {{
                text-align: center;
                color: #6c757d;
                padding: 2rem;
                background: #e9ecef;
                border-radius: 10px;
                margin-top: 3rem;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{report_title}</h1>
            <p>Reporte de Análisis OPTIBAT</p>
        </div>
        
        <div class="info-box">
            <h3>📊 Información del Análisis</h3>
            <p><strong>Dataset:</strong> {dataset_name}</p>
            <p><strong>Total de KPIs analizados:</strong> {len(selected_kpis)}</p>
            <p><strong>Registros totales:</strong> {len(df_analysis):,}</p>
            <p><strong>Variables KPI:</strong> {', '.join(selected_kpis)}</p>"""
    
    # Agregar información de filtros de fecha si están disponibles
    if start_date and end_date:
        html_content += f"""
            <p><strong>📅 Filtro de fechas aplicado:</strong> {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}</p>
            <p><strong>⏱️ Período analizado:</strong> {(end_date - start_date).days + 1} días</p>"""
    
    html_content += """
        </div>
    """
    
    # SECCIÓN DE ANÁLISIS DE EFICIENCIA GENERAL
    html_content += """
        <div class="kpi-section">
            <h2 class="kpi-title">📊 Análisis de Eficiencia General del Sistema</h2>
        """
    
    try:
        # Usar DataFrame específico para eficiencia general o fallback a df_analysis
        df_for_efficiency_calc = df_efficiency if df_efficiency is not None else df_analysis
        
        # Convertir DataFrame de Pandas a Polars para calculate_metrics
        if isinstance(df_for_efficiency_calc, pd.DataFrame):
            df_polars = pl.from_pandas(df_for_efficiency_calc)
        else:
            df_polars = df_for_efficiency_calc
            
        # Calcular métricas de eficiencia
        efficiency_metrics = calculate_metrics(df_polars)
        
        if efficiency_metrics:
            # Generar donut chart de eficiencia
            donut_fig = create_donut_chart(efficiency_metrics)
            # Incluir plotlyjs='inline' ya que es el primer gráfico del HTML
            donut_html = pio.to_html(donut_fig, include_plotlyjs='inline', div_id="efficiency_donut")
            
            # Agregar gráfico de eficiencia al HTML
            html_content += f'<div class="chart-container">{donut_html}</div>'
            
            # Agregar métricas de eficiencia como tabla
            html_content += f"""
            <div class="info-box" style="margin-top: 1rem;">
                <h4>📈 Métricas de Eficiencia</h4>
                <table style="width: 100%; border-collapse: collapse; margin-top: 1rem;">
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 0.5rem; border: 1px solid #dee2e6; font-weight: bold;">⚡ Eficiencia Actual</td>
                        <td style="padding: 0.5rem; border: 1px solid #dee2e6; color: #E31E32; font-weight: bold;">{efficiency_metrics['efficiency_percentage']:.1f}%</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem; border: 1px solid #dee2e6; font-weight: bold;">📊 Total Registros</td>
                        <td style="padding: 0.5rem; border: 1px solid #dee2e6;">{efficiency_metrics['total_records']:,}</td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 0.5rem; border: 1px solid #dee2e6; font-weight: bold;">🟢 ON & Ready</td>
                        <td style="padding: 0.5rem; border: 1px solid #dee2e6; color: #28a745;">{efficiency_metrics['breakdown']['ON & Ready']:,} min</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem; border: 1px solid #dee2e6; font-weight: bold;">🟠 OFF & Ready (Desperdiciado)</td>
                        <td style="padding: 0.5rem; border: 1px solid #dee2e6; color: #ffc107;">{efficiency_metrics['breakdown']['OFF & Ready (Desperdiciado)']:,} min</td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 0.5rem; border: 1px solid #dee2e6; font-weight: bold;">🔴 OFF & No Ready</td>
                        <td style="padding: 0.5rem; border: 1px solid #dee2e6; color: #dc3545;">{efficiency_metrics['breakdown']['OFF & No Ready']:,} min</td>
                    </tr>
                </table>
            </div>
            """
        else:
            html_content += """
            <div class="error">No se pudieron calcular las métricas de eficiencia</div>
            """
    except Exception as e:
        html_content += f"""
        <div class="error">Error calculando eficiencia general: {str(e)}</div>
        """
    
    html_content += """
        </div>
        <br><br>
    """
    
    # AGREGAR COMENTARIO SOBRE CONTEXTO OPTIBAT_READY=1
    if selected_kpis:
        html_content += """
        <div class="info-box" style="background-color: #e3f2fd; border-left: 4px solid #2196f3; margin: 1rem 0;">
            <h4 style="color: #1976d2; margin-top: 0;">🔬 Contexto de Análisis Detallado</h4>
            <p style="margin-bottom: 0;"><strong>Importante:</strong> Los siguientes análisis detallados de KPIs se realizan únicamente con datos donde <code>OPTIBAT_READY=1</code> (sistema en estado operacional).</p>
        </div>
        <br>
        """
    
    # GENERAR SECCIONES POR CADA KPI
    for i, kpi_var in enumerate(selected_kpis, 1):
        html_content += f"""
        <div class="kpi-section">
            <h2 class="kpi-title">🔍 Análisis {i}: {kpi_var}</h2>
        """
        
        try:
            charts = generate_html_charts(df_analysis, on_col, time_col, kpi_var)
            for chart in charts:
                html_content += f'<div class="chart-container">{chart}</div>'
        except Exception as e:
            html_content += f'<div class="error">Error generando análisis para {kpi_var}: {str(e)}</div>'
        
        html_content += "</div>"
    
    # FOOTER
    html_content += f"""
        <div class="footer">
            <h4>OPTIMITIVE OPTIBAT ANALYTICS SUITE</h4>
            <p>🌐 <a href="https://optimitive.com" target="_blank">optimitive.com</a></p>
            <p><strong>Developed by JC Erreguerena</strong></p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def download_html_report(html_content: str, filename: str):
    """
    Genera el botón de descarga para el reporte HTML
    """
    b64_html = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64_html}" download="{filename}">📄 Descargar Reporte HTML Completo</a>'
    return href

# ==============================================================================
# 4. INTERFAZ DE STREAMLIT (v17)
# ==============================================================================

def main():
    # Sistema de Autenticación
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {OPTIMITIVE_COLORS['primary_red']} 0%, #CC1A2C 100%);
                    color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: 900;">ACCESO RESTRINGIDO</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">OPTIMITIVE OPTIBAT ANALYTICS SUITE</p>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;">Ingrese sus credenciales para acceder</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Crear formulario de login centrado
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            """, unsafe_allow_html=True)
            
            st.markdown("### Iniciar Sesión")
            
            with st.form("login_form"):
                username = st.text_input("Usuario:", placeholder="Ingrese su usuario")
                password = st.text_input("Contraseña:", type="password", placeholder="Ingrese su contraseña")
                login_button = st.form_submit_button("INGRESAR", use_container_width=True, type="primary")
                
                if login_button:
                    if username == "OPTIMITIVE" and password == "Mantenimiento.optibat":
                        st.session_state.authenticated = True
                        st.success("¡Acceso autorizado! Redirigiendo...")
                        st.rerun()
                    else:
                        st.error("Credenciales incorrectas. Verifique usuario y contraseña.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Información de contacto
            st.markdown("""
            <div style="text-align: center; margin-top: 2rem; color: #6c757d;">
                <small><a href="https://optimitive.com" target="_blank">optimitive.com</a> | 
                Developed by JC Erreguerena</small>
            </div>
            """, unsafe_allow_html=True)
        
        return  # Terminar ejecución si no está autenticado
    
    # Header corporativo OPTIMITIVE
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {OPTIMITIVE_COLORS['primary_red']} 0%, #CC1A2C 100%);
                color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 900;">OPTIMITIVE OPTIBAT ANALYTICS</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">Advanced Performance Dashboard & Real-time Series Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # CSS para ocultar completamente la lista de archivos del file uploader + estilo corporativo
    st.markdown(f"""
    <style>
    /* Ocultar completamente todos los elementos de lista de archivos */
    .uploadedFilesList {{
        display: none !important;
    }}
    .uploadedFileName {{
        display: none !important;
    }}
    .stFileUploader > div[data-testid="stFileUploadDropzone"] + div {{
        display: none !important;
    }}
    .stFileUploader > div > div:nth-child(2) {{
        display: none !important;
    }}
    .stFileUploader div[data-testid="stFileUploaderDeleteBtn"] {{
        display: none !important;
    }}
    .stFileUploader div[class*="uploadedFile"] {{
        display: none !important;
    }}
    .stFileUploader div[class*="fileList"] {{
        display: none !important;
    }}
    [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzoneInstructions"] + div {{
        display: none !important;
    }}
    
    /* Optimitive Corporate Styling */
    .stMetric {{
        background: {OPTIMITIVE_COLORS['light_bg']};
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid {OPTIMITIVE_COLORS['border']};
    }}
    .stMetric > div > div > div:first-child {{
        color: {OPTIMITIVE_COLORS['text_secondary']};
    }}
    .stMetric > div > div > div:nth-child(2) {{
        color: {OPTIMITIVE_COLORS['primary_red']};
        font-weight: bold;
    }}
    .stSelectbox > div > div > div {{
        border-color: {OPTIMITIVE_COLORS['border']};
    }}
    .stTextInput > div > div > input {{
        border-color: {OPTIMITIVE_COLORS['border']};
    }}
    .stExpander > div > div > div:first-child {{
        background-color: {OPTIMITIVE_COLORS['medium_bg']};
        border-color: {OPTIMITIVE_COLORS['primary_red']};
    }}
    </style>
    """, unsafe_allow_html=True)

    # Inicializar session state
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = None
        st.session_state.loaded_data = None
        st.session_state.available_dataframes = {}
        st.session_state.selected_df_name = None
        st.session_state.df_filtrado = None
    
    with st.sidebar:
        st.header("1. Seleccionar Datos")
        
        # File uploader para que el usuario seleccione archivos
        uploaded_files = st.file_uploader(
            "📁 Selecciona archivos de datos:",
            type=["txt", "tsv", "csv", "osf"],
            accept_multiple_files=True,
            help="Selecciona uno o más archivos de estadísticas para analizar",
            label_visibility="visible"
        )
        
        # Carga automática cuando se seleccionan archivos
        if uploaded_files and (not st.session_state.selected_files or 
                              len(uploaded_files) != len(st.session_state.selected_files) or
                              [f.name for f in uploaded_files] != [f.name for f in st.session_state.selected_files]):
            
            # Mostrar solo contador de archivos
            st.success(f"✅ {len(uploaded_files)} archivos seleccionados")
            
            # Procesado automático con barra de progreso
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            with progress_placeholder:
                with st.spinner("Procesando archivos automáticamente..."):
                    st.session_state.loaded_data = load_data(uploaded_files)
                    st.session_state.selected_files = uploaded_files
            
            progress_placeholder.empty()
            
            if st.session_state.loaded_data:
                status_placeholder.success("✅ Datos procesados exitosamente")
                st.rerun()
            else:
                status_placeholder.error("❌ Error al procesar los archivos")
        
        elif st.session_state.selected_files and st.session_state.loaded_data:
            # Mostrar solo información esencial de archivos ya cargados
            st.success(f"✅ {len(st.session_state.selected_files)} archivos procesados")
            
            # Lista desplegable opcional con detalles de archivos
            with st.expander("📁 Ver detalles de archivos"):
                for i, file in enumerate(st.session_state.selected_files, 1):
                    file_size_mb = file.size / (1024 * 1024)
                    st.write(f"{i}. 📄 {file.name} ({file_size_mb:.2f} MB)")
                st.info(f"📊 Total: {len(st.session_state.loaded_data.df):,} registros procesados")
            
            # Botón para cargar nuevos archivos
            if st.button("🔄 Cargar Nuevos Archivos"):
                st.session_state.selected_files = None
                st.session_state.loaded_data = None
                st.rerun()
        else:
            st.info("👆 Selecciona archivos de datos para comenzar")
            return

    # Verificar si hay datos cargados
    if not st.session_state.loaded_data:
        st.info("🔍 Selecciona y procesa archivos de datos para comenzar el análisis")
        return

    df_original = st.session_state.loaded_data.df

    with st.sidebar:
        st.header("2. Filtros Globales")
        min_date, max_date = df_original['datetime'].min(), df_original['datetime'].max()
        start_date = st.date_input("Fecha de inicio", min_date.date(), min_value=min_date.date(), max_value=max_date.date())
        end_date = st.date_input("Fecha de fin", max_date.date(), min_value=min_date.date(), max_value=max_date.date())

    df_filtered = df_original.filter((pl.col('datetime') >= datetime.combine(start_date, datetime.min.time())) & (pl.col('datetime') <= datetime.combine(end_date, datetime.max.time())))

    # Selector de DataFrame para análisis
    st.header("🎯 Selección de DataFrame para Análisis")
    
    available_dfs_for_analysis = get_available_dataframes()
    
    if len(available_dfs_for_analysis) > 1:
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_analysis_df = st.selectbox(
                "Elegir DataFrame para análisis y gráficos:",
                options=list(available_dfs_for_analysis.keys()),
                key="analysis_df_selector",
                help="Este DataFrame se usará para todos los gráficos y análisis siguientes"
            )
        
        with col2:
            selected_df = available_dfs_for_analysis[selected_analysis_df]
            st.metric("Filas seleccionadas", f"{len(selected_df):,}")
            st.metric("Columnas", len(selected_df.columns))
        
        # Convertir el DataFrame seleccionado a Polars si es necesario
        if isinstance(selected_df, pd.DataFrame):
            # Asegurar que existe columna datetime
            if 'datetime' not in selected_df.columns and 'Date' in selected_df.columns:
                selected_df = selected_df.rename(columns={'Date': 'datetime'})
            df_for_analysis = pl.from_pandas(selected_df)
        else:
            df_for_analysis = selected_df
        
        # Aplicar filtros de fecha al DataFrame seleccionado
        if 'datetime' in df_for_analysis.columns:
            df_filtered = df_for_analysis.filter(
                (pl.col('datetime') >= datetime.combine(start_date, datetime.min.time())) & 
                (pl.col('datetime') <= datetime.combine(end_date, datetime.max.time()))
            )
        else:
            df_filtered = df_for_analysis
            st.warning(f"⚠️ DataFrame '{selected_analysis_df}' no tiene columna 'datetime'. Se usará sin filtros de fecha.")
        
        st.success(f"✅ Usando DataFrame: **{selected_analysis_df}** para análisis")
    else:
        # Si solo hay un DataFrame, usar el df_filtered que ya tiene filtros aplicados
        st.info("ℹ️ Solo hay un DataFrame disponible. Se usará automáticamente.")
        df_for_analysis = df_filtered

    # Donut de Eficiencia General
    st.header("📊 Análisis de Eficiencia General")
    
    # Usar el DataFrame seleccionado con filtros globales aplicados
    # Aplicar filtros de fecha al DataFrame seleccionado
    if 'datetime' in df_for_analysis.columns:
        df_for_efficiency = df_for_analysis.filter(
            (pl.col('datetime') >= datetime.combine(start_date, datetime.min.time())) & 
            (pl.col('datetime') <= datetime.combine(end_date, datetime.max.time()))
        )
    else:
        df_for_efficiency = df_for_analysis
    
    metrics = calculate_metrics(df_for_efficiency)
    if metrics: 
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(create_donut_chart(metrics), use_container_width=True)
        with col2:
            # Crear contenedor con estilo profesional para métricas
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border: 1px solid #dee2e6;">
                <h3 style="margin-top: 0; color: #1f2937; font-size: 1.3rem;">📊 Métricas de Rendimiento</h3>
            """, unsafe_allow_html=True)
            
            # Eficiencia con formato destacado
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">⚡ Eficiencia Actual</p>
                <p style="margin: 0; color: #E31E32; font-size: 2rem; font-weight: bold;">{metrics['efficiency_percentage']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Total de registros
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">📊 Total Registros</p>
                <p style="margin: 0; color: #1f2937; font-size: 1.5rem; font-weight: bold;">{metrics['total_records']:,}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if metrics['breakdown']:
                # ON & Ready
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">🟢 ON & Ready</p>
                    <p style="margin: 0; color: #28a745; font-size: 1.3rem; font-weight: bold;">{metrics['breakdown']['ON & Ready']:,} min</p>
                </div>
                """, unsafe_allow_html=True)
                
                # OFF & Ready
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">🟠 OFF & Ready (Desperdiciado)</p>
                    <p style="margin: 0; color: #ffc107; font-size: 1.3rem; font-weight: bold;">{metrics['breakdown']['OFF & Ready (Desperdiciado)']:,} min</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")

    # NUEVA SECCIÓN: Serie Temporal Automática con Selector Arriba
    st.header("📈 Serie Temporal OPTIBAT - Análisis Interactivo")
    
    # Selector de variables ARRIBA del gráfico
    st.subheader("🔧 Selección de Variables")
    selected_vars, binary_vars, line_styles = create_variable_selector(df_for_efficiency)
    
    # Gráfico de serie temporal DEBAJO ocupando TODO EL ANCHO
    st.subheader("📊 Visualización Temporal Completa")
    
    # Siempre mostrar el gráfico, incluso sin variables seleccionadas
    if len(df_for_efficiency) > 0:
        # Si no hay variables seleccionadas, mostrar solo OPTIBAT_ON
        if not selected_vars:
            # Crear gráfico básico con solo OPTIBAT_ON
            basic_vars = {"OPTIBAT_ON": ColorScheme.NEUTRAL_BLUE}
            basic_binary = {"OPTIBAT_ON": True}
            basic_styles = {"OPTIBAT_ON": "solid"}
            st.plotly_chart(create_time_series_chart(df_for_efficiency, basic_vars, basic_binary, basic_styles), use_container_width=True)
            st.info("**Serie temporal básica OPTIBAT_ON mostrada.** Selecciona variables adicionales arriba para análisis completo.")
        else:
            # Mostrar con variables seleccionadas
            st.plotly_chart(create_time_series_chart(df_for_efficiency, selected_vars, binary_vars, line_styles), use_container_width=True)
    else:
        st.error("❌ No hay datos para mostrar en el rango de fechas seleccionado")

    # Sección de estadísticas descriptivas
    if selected_vars:
        st.header("📊 Estadísticas Descriptivas de Variables Seleccionadas")
        selected_kpis = list(selected_vars.keys())
        create_statistics_table(df_for_efficiency, selected_kpis)

    st.markdown("---")
    
    # Sección de filtrado avanzado de DataFrames
    render_dataframe_filter_ui()
    
    st.markdown("---")
    
    # NUEVA SECCIÓN: Análisis Detallado de KPIs (ON vs OFF) - DESPUÉS DEL FILTRADO
    st.header("📈 Análisis Detallado de KPIs (ON vs OFF)")
    st.info("🔬 **Contexto de Análisis**: Este análisis se realiza únicamente con datos donde OPTIBAT_READY=1 (sistema operacional)")
    
    # SELECTORES DE CONTROL
    col_selector1, col_selector2 = st.columns([1, 1])
    
    with col_selector1:
        # 1. SELECTOR DE DATAFRAME
        available_dfs_for_kpi = get_available_dataframes()
        if len(available_dfs_for_kpi) > 1:
            selected_df_for_kpi = st.selectbox(
                "📊 Selecciona DataFrame para análisis:",
                options=list(available_dfs_for_kpi.keys()),
                key="kpi_analysis_df_selector",
                help="Elige el DataFrame que quieres analizar en detalle"
            )
            df_for_kpi_analysis = available_dfs_for_kpi[selected_df_for_kpi]
            st.success(f"✅ Usando DataFrame: **{selected_df_for_kpi}** ({len(df_for_kpi_analysis):,} filas)")
        else:
            # Si solo hay un DataFrame, usar el actual
            df_for_kpi_analysis = df_filtered.to_pandas()
            st.info("📊 Usando DataFrame principal automáticamente")
    
    with col_selector2:
        # 2. SELECTOR DE VARIABLE KPI
        # Detectar todas las variables numéricas disponibles (excluyendo binarias y datetime)
        if isinstance(df_for_kpi_analysis, pd.DataFrame):
            numeric_cols = df_for_kpi_analysis.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Si es Polars DataFrame, convertir a pandas
            df_for_kpi_analysis = df_for_kpi_analysis.to_pandas()
            numeric_cols = df_for_kpi_analysis.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filtrar columnas que no son binarias ni de control
        exclude_cols = ['OPTIBAT_ON', 'OPTIBAT_READY', 'datetime']
        kpi_options = [col for col in numeric_cols if col not in exclude_cols]
        
        if kpi_options:
            selected_kpis_for_detail = st.multiselect(
                "🎯 Selecciona Variables KPI:",
                options=kpi_options,
                help="Selecciona una o múltiples variables. Se generará análisis completo para cada una.",
                key="detailed_kpi_variables_selector"
            )
            st.metric("Variables disponibles", len(kpi_options))
            if selected_kpis_for_detail:
                st.success(f"✅ {len(selected_kpis_for_detail)} variable(s) seleccionada(s)")
        else:
            selected_kpis_for_detail = []
            st.warning("⚠️ No hay variables KPI disponibles")
    
    # ANÁLISIS DETALLADO MÚLTIPLE - POR CADA KPI SELECCIONADO
    if selected_kpis_for_detail and len(available_dfs_for_kpi) > 0:
        # Detectar columna de tiempo primero
        time_col = None
        if 'datetime' in df_for_kpi_analysis.columns:
            time_col = 'datetime'
        else:
            # Buscar otra columna de tiempo
            time_cols = [c for c in df_for_kpi_analysis.columns if any(t in c.lower() for t in ['date', 'time', 'fecha'])]
            if time_cols:
                time_col = time_cols[0]
        
        # APLICAR FILTROS GLOBALES AL DATAFRAME DE ANÁLISIS KPI
        if time_col and time_col in df_for_kpi_analysis.columns:
            # Convertir fechas para comparación
            df_for_kpi_analysis[time_col] = pd.to_datetime(df_for_kpi_analysis[time_col], errors='coerce')
            
            # Aplicar filtro de fechas globales
            mask = (df_for_kpi_analysis[time_col] >= pd.Timestamp.combine(start_date, datetime.min.time())) & \
                   (df_for_kpi_analysis[time_col] <= pd.Timestamp.combine(end_date, datetime.max.time()))
            
            df_for_kpi_analysis_filtered = df_for_kpi_analysis[mask].copy()
            
            # Mostrar información del filtrado
            st.info(f"📅 Filtro de fechas aplicado: {start_date} a {end_date} | Registros: {len(df_for_kpi_analysis_filtered):,} de {len(df_for_kpi_analysis):,}")
        else:
            df_for_kpi_analysis_filtered = df_for_kpi_analysis
            st.warning("⚠️ No se pudo aplicar el filtro de fechas al DataFrame de KPIs (columna de tiempo no encontrada)")
        
        # Usar el DataFrame filtrado para el análisis
        df_for_kpi_analysis = df_for_kpi_analysis_filtered
        
        # Detectar columna ON para análisis detallado
        on_col_for_analysis = "OPTIBAT_ON"
        if on_col_for_analysis not in df_for_kpi_analysis.columns:
            # Buscar alternativa
            possible_on_cols = [c for c in df_for_kpi_analysis.columns if "on" in c.lower()]
            if possible_on_cols:
                on_col_for_analysis = possible_on_cols[0]
        
        # Verificar que encontramos la columna ON
        if on_col_for_analysis in df_for_kpi_analysis.columns and time_col:
            # ANÁLISIS POR CADA KPI SELECCIONADO
            for i, selected_kpi in enumerate(selected_kpis_for_detail, 1):
                
                # Encabezado del KPI
                st.markdown(f"## 🔍 Análisis {i}: {selected_kpi}")
                
                # Mostrar información del DataFrame seleccionado
                if len(available_dfs_for_kpi) > 1:
                    st.info(f"📊 DataFrame: **{selected_df_for_kpi}** | Registros: {len(df_for_kpi_analysis):,} | Variable: **{selected_kpi}**")
                
                # VISUALIZACIÓN ESTILO INFORME HTML - Layout en bloque completo
                
                # 1. Time Series Continua - ANCHO COMPLETO
                st.markdown(f"### 📈 {selected_kpi} - Serie Temporal Continua (ON vs OFF)")
                try:
                    ts_fig = ts_on_off_with_means(df_for_kpi_analysis, time_col, on_col_for_analysis, selected_kpi)
                    st.plotly_chart(ts_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creando time series para {selected_kpi}: {e}")
                
                # 2. Boxplot - ANCHO COMPLETO
                st.markdown(f"### 📊 {selected_kpi} - Boxplot ON vs OFF")
                try:
                    box_fig = create_boxplot_on_off(df_for_kpi_analysis, on_col_for_analysis, selected_kpi)
                    st.plotly_chart(box_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creando boxplot para {selected_kpi}: {e}")
                
                # 3. Distribución - ANCHO COMPLETO
                st.markdown(f"### 📈 {selected_kpi} - Distribución ON vs OFF")
                try:
                    dist_fig = create_distribution_plot(df_for_kpi_analysis, on_col_for_analysis, selected_kpi)
                    st.plotly_chart(dist_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creando distribución para {selected_kpi}: {e}")
                
                # 4. Tabla de Estadísticas Detalladas - ANCHO COMPLETO
                st.markdown(f"### 📋 {selected_kpi} - Estadísticas Detalladas")
                try:
                    stats_fig = create_statistics_table_detailed(df_for_kpi_analysis, on_col_for_analysis, selected_kpi)
                    st.plotly_chart(stats_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creando estadísticas para {selected_kpi}: {e}")
                
                # Separador entre KPIs (excepto el último)
                if i < len(selected_kpis_for_detail):
                    st.markdown("---")
                    st.markdown("<br>", unsafe_allow_html=True)
        else:
            if not time_col:
                st.warning("⚠️ No se encontró columna de tiempo válida para análisis temporal")
            else:
                st.warning(f"⚠️ No se encontró la columna '{on_col_for_analysis}' necesaria para el análisis ON vs OFF")
    else:
        st.info("ℹ️ Selecciona al menos un KPI para ver el análisis detallado")

    # NUEVA SECCIÓN: Exportación HTML del Análisis Completo
    if selected_kpis_for_detail and len(available_dfs_for_kpi) > 0:
        st.markdown("---")
        st.header("📄 Exportar Reporte HTML Completo")
        
        # Campo para título personalizable
        col_title, col_dataset = st.columns([2, 1])
        
        with col_title:
            report_title = st.text_input(
                "📝 Título del Reporte:",
                value=f"Análisis OPTIBAT - {datetime.now().strftime('%d/%m/%Y')}",
                help="Personaliza el título que aparecerá en el reporte HTML",
                key="html_report_title"
            )
        
        with col_dataset:
            # Detectar nombre del dataset
            if len(available_dfs_for_kpi) > 1:
                dataset_display_name = selected_df_for_kpi
            else:
                dataset_display_name = "DataFrame Principal"
            st.info(f"📊 Dataset: {dataset_display_name}")
        
        # Botón para generar y descargar HTML
        col_btn, col_info = st.columns([1, 1])
        
        with col_btn:
            if st.button("🚀 GENERAR REPORTE HTML COMPLETO", use_container_width=True, type="primary"):
                with st.spinner("📊 Generando reporte HTML con todos los gráficos..."):
                    try:
                        # Construir DataFrame para análisis de eficiencia general del HTML
                        # Usar el mismo DataFrame que el usuario seleccionó para visualización
                        if len(available_dfs_for_analysis) > 1:
                            # Usar el DataFrame seleccionado en "Selección de DataFrame para Análisis"
                            selected_df_for_html = available_dfs_for_analysis[selected_analysis_df]
                            if isinstance(selected_df_for_html, pd.DataFrame):
                                df_for_html_base = pl.from_pandas(selected_df_for_html)
                            else:
                                df_for_html_base = selected_df_for_html
                        else:
                            # Usar el DataFrame original con filtro
                            df_for_html_base = df_original
                        
                        # Aplicar filtros de fecha globales
                        if 'datetime' in df_for_html_base.columns:
                            df_for_html_efficiency = df_for_html_base.filter(
                                (pl.col('datetime') >= datetime.combine(start_date, datetime.min.time())) & 
                                (pl.col('datetime') <= datetime.combine(end_date, datetime.max.time()))
                            ).to_pandas()
                        else:
                            df_for_html_efficiency = df_for_html_base.to_pandas()
                        
                        html_content = create_html_report(
                            df_analysis=df_for_kpi_analysis,  # DataFrame específico para análisis KPIs
                            selected_kpis=selected_kpis_for_detail,
                            on_col=on_col_for_analysis,
                            time_col=time_col,
                            report_title=report_title,
                            dataset_name=dataset_display_name,
                            start_date=start_date,
                            end_date=end_date,
                            df_efficiency=df_for_html_efficiency  # DataFrame específico para análisis eficiencia
                        )
                        
                        # Generar nombre de archivo
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_title = re.sub(r'[^\w\s-]', '', report_title).strip()[:50]
                        safe_title = re.sub(r'[-\s]+', '_', safe_title)
                        filename = f"OPTIBAT_Reporte_{safe_title}_{timestamp}.html"
                        
                        # Crear botón de descarga
                        st.success("✅ Reporte HTML generado exitosamente!")
                        
                        # Convertir a bytes para descarga
                        html_bytes = html_content.encode('utf-8')
                        
                        # Botón de descarga
                        download_clicked = st.download_button(
                            label="📄 DESCARGAR REPORTE HTML COMPLETO",
                            data=html_bytes,
                            file_name=filename,
                            mime="text/html",
                            use_container_width=True,
                            key="download_html_report"
                        )
                        
                        # Auto-click del botón de descarga usando JavaScript
                        st.markdown("""
                        <script>
                        setTimeout(function() {
                            const downloadButton = document.querySelector('[data-testid="stDownloadButton"] button');
                            if (downloadButton) {
                                downloadButton.click();
                            }
                        }, 500);
                        </script>
                        """, unsafe_allow_html=True)
                        
                        # Información adicional
                        st.info(f"📁 Archivo: **{filename}** | Tamaño: {len(html_bytes) / 1024:.1f} KB | KPIs: {len(selected_kpis_for_detail)}")
                        
                    except Exception as e:
                        st.error(f"❌ Error generando reporte HTML: {str(e)}")
        
        with col_info:
            st.info(f"""
            **📊 El reporte incluirá:**
            • {len(selected_kpis_for_detail)} análisis de KPI completos
            • Serie temporal, boxplot, distribución y estadísticas por cada KPI
            • Gráficos interactivos Plotly embebidos
            • Filtrado OPTIBAT_READY=1 aplicado
            • Filtro de fechas: {start_date} a {end_date}
            • Diseño profesional Optimitive
            """)

    st.markdown("---")
    
    # Sección de Descarga con Estilo Corporativo
    st.header("💾 Exportación de Datos")
    
    col1, col2 = st.columns(2)
    with col1:
        buffer = io.BytesIO()
        # Generar TSV usando pandas para compatibilidad con Polars
        tsv_string = df_filtered.to_pandas().to_csv(index=False, sep='\t')
        buffer.write(tsv_string.encode('utf-8'))
        buffer.seek(0)
        st.download_button(
            label="📊 Descargar TSV Filtrado", 
            data=buffer, 
            file_name=f"OPTIBAT_datos_filtrados_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.tsv", 
            mime="text/tab-separated-values",
            use_container_width=True
        )
    
    with col2:
        st.info(f"📈 Datos preparados: {len(df_filtered):,} registros del {start_date} al {end_date}")

    # Footer Corporativo OPTIMITIVE
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: {OPTIMITIVE_COLORS['text_secondary']}; 
                padding: 2rem; background: {OPTIMITIVE_COLORS['medium_bg']}; 
                border-radius: 10px; margin-top: 2rem;">
        <h4 style="color: {OPTIMITIVE_COLORS['primary_red']};">OPTIMITIVE OPTIBAT ANALYTICS SUITE</h4>
        <p><strong>AI Optimization Solutions</strong></p>
        <p>🌐 <a href="https://optimitive.com" target="_blank" style="color: {OPTIMITIVE_COLORS['primary_red']};">optimitive.com</a></p>
        <p><strong>Developed by JC Erreguerena.</strong>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid {OPTIMITIVE_COLORS['border']};">
            <small>🔋 Professional OPTIBAT System Monitoring & Advanced Time Series Analytics</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()