import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from pathlib import Path
import plotly.express as px
from geopy.distance import geodesic


# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent

# Coordenadas del centro de Ámsterdam (Dam Square)
CENTRO_AMSTERDAM = (52.373079, 4.892453)

# Configuración de la página
st.set_page_config(
    page_title="🏠 Predicción Precios Ámsterdam",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🏠 Predicción y Exploración de Precios en Ámsterdam")
st.markdown("""
Explora propiedades y predice precios en diferentes zonas de Ámsterdam.
**Zona 0** usa un modelo especializado, el resto de zonas usan el modelo general.
""")

# Cargar modelos
@st.cache_resource
def cargar_modelos():
    try:
        modelos_dir = BASE_DIR / "models"
        return {
            'zona_0': joblib.load(modelos_dir / "modelo_zona_0_especial.pkl"),
            'general': joblib.load(modelos_dir / "modelo_general.pkl"),
            'kmeans': joblib.load(modelos_dir / "kmeans_model.pkl"),
            'caracteristicas': joblib.load(modelos_dir / "caracteristicas.pkl")
        }
    except Exception as e:
        st.error(f"Error cargando modelos: {str(e)}")
        st.stop()

# Cargar datos
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv(BASE_DIR / "data" / 'datos_cluster.csv')
        # Calcular distancia al centro para cada propiedad
        df['Distancia_Centro'] = df.apply(lambda row: geodesic(
            (row['Lat'], row['Lon']), 
            CENTRO_AMSTERDAM
        ).km, axis=1)
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        st.stop()

try:
    modelos = cargar_modelos()
    df_cluster = cargar_datos()
    st.sidebar.success("✅ Modelos y datos cargados correctamente")
except Exception as e:
    st.error(f"❌ Error cargando recursos: {str(e)}")
    st.stop()

# Obtener centroides de zonas
centroides = df_cluster.groupby('Zona')[['Lat', 'Lon']].mean().reset_index()

# Mapeo de colores por zona
colores_zonas = {
    0: "red",     # Centro
    1: "green",   # Sureste
    2: "purple",  # Oeste
    3: "blue",    # Este
    4: "orange"   # Noreste
}

# ===========================================
# SECCIÓN 1: EXPLORACIÓN DE PROPIEDADES
# ===========================================
st.header("🔍 Exploración de Propiedades")

# Filtros en sidebar
with st.sidebar:
    st.header("🎛️ Filtros de Exploración")
    
    # Opción para seleccionar todas las zonas
    todas_zonas = st.checkbox("Seleccionar todas las zonas", value=True)
    
    # Filtro por zona
    zonas_disponibles = sorted(df_cluster['Zona'].unique())
    zonas = st.multiselect(
        "Seleccionar zonas:", 
        options=zonas_disponibles,
        default=zonas_disponibles if todas_zonas else [0]
    )
    
    # Actualizar selección si se marca/desmarca "todas las zonas"
    if todas_zonas:
        zonas = zonas_disponibles
    
    # Filtro por habitaciones
    habitaciones = st.multiselect(
        "Número de habitaciones:", 
        options=sorted(df_cluster['Room'].unique()),
        default=sorted(df_cluster['Room'].unique())
    )
    
    # Filtro por área
    min_area = int(df_cluster['Area'].min())
    max_area = int(df_cluster['Area'].max())
    rango_area = st.slider(
        "Rango de área (m²):",
        min_value=min_area,
        max_value=max_area,
        value=(min_area, max_area)
    )

# Aplicar filtros
df_filtrado = df_cluster[
    (df_cluster['Zona'].isin(zonas)) &
    (df_cluster['Room'].isin(habitaciones)) &
    (df_cluster['Area'] >= rango_area[0]) &
    (df_cluster['Area'] <= rango_area[1])
]

# Mostrar estadísticas
st.subheader(f"📊 Propiedades encontradas: {len(df_filtrado)}")
if not df_filtrado.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("Precio Promedio", f"{df_filtrado['Price'].mean():,.0f} €")
    col2.metric("Área Promedio", f"{df_filtrado['Area'].mean():.0f} m²")
    col3.metric("Habitaciones Promedio", f"{df_filtrado["Room"].mean():.0f}")
   
# Mapa interactivo de propiedades
st.subheader("🗺️ Mapa de Propiedades")
st.caption("Selecciona una propiedad para ver detalles")

# Crear mapa centrado en Ámsterdam
m = folium.Map(location=[52.370216, 4.895168], zoom_start=11.2)

# Añadir propiedades filtradas con colores consistentes
for _, row in df_filtrado.iterrows():
    zona = row['Zona']
    color = colores_zonas.get(zona, 'gray')  # Usar gris si la zona no está en el mapeo
    
    folium.Marker(
        location=[row['Lat'], row['Lon']],
        popup=(
            f"💵 <b>Precio:</b> €{row['Price']:,.0f}<br>"
            f"📐 <b>Área:</b> {row['Area']} m²<br>"
            f"🚪 <b>Habitaciones:</b> {int(row['Room'])}<br>"
            f"🗺️ <b>Zona:</b> {int(zona)}<br>"
            
        ),
        tooltip=f"💵 {row['Price']:,.0f}€ | 📐 {row['Area']}m² | 🚪 {int(row['Room'])} hab | 🗺️ Zona {int(zona)}",
        icon=folium.Icon(color=color)
    ).add_to(m)

# Mostrar mapa en Streamlit
mapa_evento = st_folium(m, width=1000, height=500)

# Mostrar detalles de propiedad seleccionada
if mapa_evento.get("last_object_clicked"):
    clicked_location = mapa_evento["last_object_clicked"]
    lat_click = clicked_location["lat"]
    lon_click = clicked_location["lng"]
    
    # Encontrar propiedad más cercana al click
    df_filtrado['distancia'] = np.sqrt(
        (df_filtrado['Lat'] - lat_click)**2 + 
        (df_filtrado['Lon'] - lon_click)**2
    )
    propiedad_seleccionada = df_filtrado.loc[df_filtrado['distancia'].idxmin()]
    
    st.divider()
    st.subheader("🏡 Propiedad Seleccionada")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Precio", f"€{propiedad_seleccionada['Price']:,.0f}")
        st.metric("Área", f"{propiedad_seleccionada['Area']} m²")
        st.metric("Habitaciones", propiedad_seleccionada['Room'])
        st.metric("Zona", propiedad_seleccionada['Zona'])
        
    
    with col2:
        # Mini mapa de la ubicación exacta
        m_mini = folium.Map(location=[lat_click, lon_click], zoom_start=15)
        folium.Marker(
            [lat_click, lon_click],
            popup="Propiedad seleccionada",
            icon=folium.Icon(color='green')
        ).add_to(m_mini)
        st_folium(m_mini, width=1600, height=800)


# ===========================================
# SECCIÓN 2: PREDICCIÓN DE PRECIOS
# ===========================================
st.divider()
st.header("🔮 Predicción de Precios")

# Crear dos columnas
col_izq, col_der = st.columns(2)

with col_izq:
    st.subheader("Características de la Propiedad")
    
    # Inputs básicos
    area = st.slider("Área (m²)", 30, 500, 80, key='area_pred')
    habitaciones = st.slider("Habitaciones", 1, 10, 2, key='habitaciones_pred')
    zona_seleccionada = st.selectbox("Zona:", options=[0, 1, 2, 3, 4], key='zona_pred')
    
    # Botón para calcular
    calcular_btn = st.button("Calcular Precio", type="primary", key='calcular_btn')

# Obtener centroide de la zona seleccionada
centroide = centroides[centroides['Zona'] == zona_seleccionada].iloc[0]
lat = centroide['Lat']
lon = centroide['Lon']

# Inicializar variable de predicción
prediccion = None

# Calcular predicción solo cuando se presiona el botón
if calcular_btn:
    try:
        # Preparar datos de entrada
        room_area = area / habitaciones if habitaciones > 0 else area
        
        # Manejar características específicas para cada modelo
        if zona_seleccionada == 0:
            # Características ESPECÍFICAS
            caracteristicas_esperadas = ['Area', 'Room', 'Lat', 'Lon', 'Room_Area']
            datos_propiedad = {
                'Area': area,
                'Room': habitaciones,
                'Lat': lat,
                'Lon': lon,
                'Room_Area': room_area
            }
        else:
            # Características para el modelo general
            caracteristicas_esperadas = modelos['caracteristicas']
            datos_propiedad = {
                'Area': area,
                'Room': habitaciones,
                'Room_Area': room_area,
                'Lon': lon,
                'Lat': lat
            }
        
        # Crear array de valores
        valores_ordenados = [datos_propiedad[feature] for feature in caracteristicas_esperadas]
        
        # Crear DataFrame con las columnas
        input_data = pd.DataFrame([valores_ordenados], columns=caracteristicas_esperadas)
        
        # Seleccionar modelo según la zona
        if zona_seleccionada == 0:
            modelo_usado = modelos['zona_0']
            modelo_nombre = "Modelo Especial Zona 0"
            color = "red"
        else:
            modelo_usado = modelos['general']
            modelo_nombre = "Modelo General"
            color = colores_zonas.get(zona_seleccionada, 'blue')
        
        # Predecir precio
        precio = modelo_usado.predict(input_data)[0]
        
        prediccion = {
            'precio': precio,
            'modelo_nombre': modelo_nombre,
            'zona_seleccionada': zona_seleccionada,
            'area': area,
            'habitaciones': habitaciones,
            'lat': lat,
            'lon': lon,
            'color': color
        }
        
        # Almacenar en sesión para persistencia
        st.session_state.prediccion = prediccion
        
    except Exception as e:
        st.error(f"Error en predicción: {str(e)}")
        st.stop()

# Si ya existe una predicción en sesión
if 'prediccion' in st.session_state and not calcular_btn:
    prediccion = st.session_state.prediccion

# Mostrar resultados si existe predicción
if prediccion:
    with col_der:
        st.subheader("📊 Resultado de Predicción")
        st.metric("Precio Estimado", f"€{prediccion['precio']:,.2f}", delta_color="off")
        
        st.subheader("Características de Entrada")
        st.write(f"**Área:** {prediccion['area']} m²")
        st.write(f"**Habitaciones:** {prediccion['habitaciones']}")
        st.write(f"**Zona:** {prediccion['zona_seleccionada']}")
        
        st.subheader("🗺️ Ubicación Estimada (Centroide de Zona)")
        m_pred = folium.Map(location=[prediccion['lat'], prediccion['lon']], zoom_start=14)
        folium.Marker(
            [prediccion['lat'], prediccion['lon']],
            popup=f"Precio estimado: €{prediccion['precio']:,.2f}",
            tooltip=f"Zona {prediccion['zona_seleccionada']}",
            icon=folium.Icon(color=prediccion['color'], icon="home")
        ).add_to(m_pred)
        st_folium(m_pred, width=500, height=300)
        
        st.caption("🔴 Zona 0 (Centro) | 🟢 Zona 1 (Sureste) | 🟣 Zona 2 (Oeste) | 🔵 Zona 3 (Este) | 🟠 Zona 4 (Noreste)")
else:
    with col_der:
        st.info("Ingrese los datos de la propiedad y haga clic en 'Calcular Precio' para obtener una predicción")


# ===========================================
# SECCIÓN 3: ANÁLISIS DE ZONAS
# ===========================================
st.divider()
st.header("📈 Análisis por Zonas")

# Gráfico de precios por zona
st.subheader("Precio Promedio por Zona")
fig = px.bar(
    df_cluster.groupby('Zona')['Price'].mean().reset_index(),
    x='Zona',
    y='Price',
    color='Zona',
    color_discrete_map=colores_zonas,
    labels={'Price': 'Precio Promedio (€)', 'Zona': 'Zona'},
    text_auto='.2s'
)
st.plotly_chart(fig, use_container_width=True)


# Footer
st.divider()
st.caption("© 2025 - Sistema de Predicción de Precios de Viviendas en Ámsterdam | Desarrollado con Streamlit")