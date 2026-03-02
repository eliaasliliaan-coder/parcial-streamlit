# Carga de Librerias -----------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

# Configuración de página
st.set_page_config(page_title="Dashboard Económico", layout="wide")
# -------------------------------------------------------------------------------

# Estilo Harvard Growth Lab -----------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: white;
    }
    h1 {
        color: #0B3D91;
        font-family: 'Helvetica', sans-serif;
        font-weight: 700;
        font-size: 48px;
        text-align: center;
    }
    h2 {
        color: #2E2E2E;
        font-family: 'Helvetica', sans-serif;
        font-weight: 600;
        font-size: 32px;
        text-align: center;
    }
    h3 {
        color: #6B6B6B;
        font-family: 'Helvetica', sans-serif;
        font-weight: 500;
        font-size: 24px;
        text-align: center;
    }

     h4 {
        color: #6B6B6B;
        font-family: 'Helvetica', sans-serif;
        font-weight: 500;
        font-size: 24px;
        text-align: left;
    }
    
    p {
        font-family: 'Helvetica', sans-serif;
        font-size: 18px;
        text-align: justify;
    }
    </style>
""", unsafe_allow_html=True)

#-----------------------------------------------------------------------------------

# Títulos ---------------------------------------------------------------------------
st.markdown("<h1>PARCIAL UNO - PROYECTO</h1>", unsafe_allow_html=True)
st.markdown("<h2>Del Dato al Futuro: Validación econométrica de metodologías para pronosticar remesas en Guatemala</h2>", unsafe_allow_html=True)
#------------------------------------------------------------------------------------

# Leer archivo CSV -------------------------------------------------------------------
archivo = 'Remesas2002_2026.csv'
st.set_page_config(page_title="Dashboard Económico", layout="centered")

# Cargar datos automáticamente
datos = pd.read_csv(archivo, sep=';')
st.markdown("**Vista previa de los Datos:**")
# Cargar datos automáticamente datos = pd.read_csv(archivo, sep=';') st.markdown("**Vista previa de los Datos:**") 
st.dataframe(datos) # Convertir mes texto a número mes_dict = { "Enero":1,"Febrero":2,"Marzo":3,"Abril":4,"Mayo":5,"Junio":6, "Julio":7,"Agosto":8,"Septiembre":9,"Octubre":10,"Noviembre":11,"Diciembre":12 } 
datos["Mes_num"] = datos["Mes"].map(mes_dict) 
# Crear columna fecha real datos["Fecha"] = 
pd.to_datetime( dict(year=datos["Ano"], month=datos["Mes_num"], day=1) ) datos = datos.sort_values("Fecha")

# Convertir mes texto a número
mes_dict = {
    "Enero":1,"Febrero":2,"Marzo":3,"Abril":4,"Mayo":5,"Junio":6,
    "Julio":7,"Agosto":8,"Septiembre":9,"Octubre":10,"Noviembre":11,"Diciembre":12
}

datos["Mes_num"] = datos["Mes"].map(mes_dict)

# Crear columna fecha real
datos["Fecha"] = pd.to_datetime(
    dict(year=datos["Ano"],
         month=datos["Mes_num"],
         day=1)
)

datos = datos.sort_values("Fecha")
# ------------------------------------------------------------------------------------------

# Filtro -----------------------------------------------------------------------------------
st.sidebar.header("Filtros")

# Filtro de Año (rango)
anio_min = int(datos["Ano"].min())
anio_max = int(datos["Ano"].max())

rango_anio = st.sidebar.slider(
    "Selecciona el rango de años:",
    min_value=anio_min,
    max_value=anio_max,
    value=(anio_min, anio_max)
)

# Filtro de Mes (selección múltiple)
meses_unicos = list(mes_dict.keys())

meses_seleccionados = st.sidebar.subheader("Filtrar por Mes")

# Inicializar estado si no existe
if "meses" not in st.session_state:
    st.session_state.meses = {mes: True for mes in mes_dict.keys()}

# Botón seleccionar todos
if st.sidebar.button("Seleccionar todos"):
    for mes in st.session_state.meses:
        st.session_state.meses[mes] = True

# Botón deseleccionar todos
if st.sidebar.button("Deseleccionar todos"):
    for mes in st.session_state.meses:
        st.session_state.meses[mes] = False

# Crear checkboxes individuales
meses_seleccionados = []

for mes in mes_dict.keys():
    st.session_state.meses[mes] = st.sidebar.checkbox(
        mes,
        value=st.session_state.meses[mes]
    )
    
    if st.session_state.meses[mes]:
        meses_seleccionados.append(mes)

# Aplicar filtros
datos_filtrados = datos[
    (datos["Ano"] >= rango_anio[0]) &
    (datos["Ano"] <= rango_anio[1]) &
    (datos["Mes"].isin(meses_seleccionados))
]
# -------------------------------------------------------------------------------------

# Gráfico de Divisas 2002-2026 ----------------------------------------------------------
fig = px.line(
    datos_filtrados,
    x="Fecha",
    y="Divisas",
    markers=True,
    template="plotly_white",
    title="Remesas de Guatemala para el año 2002 - 2026",
)

fig.update_layout(
    title=f"Remesas en Guatemala ({rango_anio[0]} - {rango_anio[1]})",
    xaxis_title="Año",
    yaxis_title="Divisas (Millones USD)",
    title_x=0.5,
    title_font=dict(size=20),
    height=650
)

fig.update_traces(marker=dict(size=4, color="#4A0099"),
    line=dict(width=1.5, color="#4A0099"))
st.plotly_chart(fig, use_container_width=True)
# ---------------------------------------------------------------------------------------------

# Modelo de Datos Originales ------------------------------------------------------------------
st.subheader("Modelo de Regresión Lineal - Datos Originales")

# Variable dependiente
y = datos['Divisas'].values

# Parámetros
codi = 1
pronostico = 12
codfi = len(y) + 1

# Codificación t
t = np.arange(codi, len(y) + 1).reshape(-1, 1)

# Modelo
modelo = LinearRegression()
modelo.fit(t, y)

# Predicción sobre datos históricos (tendencia)
y_pred = modelo.predict(t)

# Valores futuros
t_futuro = np.arange(codfi, codfi + pronostico).reshape(-1, 1)
pronosticos = modelo.predict(t_futuro)

# Tabla de pronóstico de Datos Originales -------------------------------------------------------
tabla_pronostico = pd.DataFrame({
    't': t_futuro.flatten(),
    'Pronostico': pronosticos
})

tabla_pronostico["Pronostico"] = tabla_pronostico["Pronostico"].round(2)

# Estilo profesional
styled_table = (
    tabla_pronostico.style
    .background_gradient(cmap="Purples", subset=["Pronostico"])
    .set_properties(**{
        'border': '1px solid #000000',
        'text-align': 'center',
        'font-size': '14px'
    })
    .set_table_styles([
        {'selector': 'th',
         'props': [('background-color', '##4A0099'),
                   ('color', 'white'),
                   ('font-weight', 'bold'),
                   ('text-align', 'center')]}
    ])
)

st.markdown("**Tabla de Pronósticos:**")
st.dataframe(styled_table, use_container_width=True)

# Gráfica de Datos Originales ----------------------------

fig = go.Figure()
import plotly.graph_objects as go

# Datos originales
fig.add_trace(go.Scatter(
    x=t.flatten(),
    y=y,
    mode='lines+markers',
    name='Datos Originales',
    line=dict(width=1.5, color="#4A0099"),         
    marker=dict(size=4, color='#4A0099'),
    
))

# Tendencia
fig.add_trace(go.Scatter(
    x=t.flatten(),
    y=y_pred,
    mode='lines',
    name='Tendencia (Regresión)', 
    line=dict(color='red'),         
    marker=dict(size=4,color='red')
))

# Pronóstico
fig.add_trace(go.Scatter(
    x=t_futuro.flatten(),
    y=pronosticos,
    mode='markers',
    name='Pronóstico',
    marker=dict(size=4,color='red')
))

fig.update_layout(
    title="Gráfica del Pronóstico de las Remesas de Guatemala del año 2002 al 2024 con Datos Originales (Millones de USD)",
    xaxis_title="Periodo (t)",
    yaxis_title="Divisas",
    hovermode="x unified",
    template="plotly_white",
    title_font=dict(size=20),
    height=650
)

st.plotly_chart(fig, use_container_width=True)
# ----------------------------------------------------------------------------------------------------------------

