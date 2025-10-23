import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from geopy.distance import distance #Representación Amsterdam
import folium #Mapas
from folium.plugins import HeatMap
import numpy as np
from sklearn.cluster import KMeans

df_house_cluster = pd.read_csv("datos_cluster.csv")

# --------------------------------------------
#               Mapa por cluster
# --------------------------------------------

center_coords = (52.377956, 4.897070) # Lon y Lat de Amsterdam
df_house_cluster['dist_to_center'] = df_house_cluster.apply(lambda row: distance(center_coords, (row["Lat"], row["Lon"])).km, axis=1)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df_house_cluster[['Lat', 'Lon']])

center_cluster = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=["Lat", "Lon"]
)

n_zona = {
    0: "Centro",
    1: "Sureste",
    2: "Oeste",
    3: "Este",
    4: "Noreste"
}

df_house_cluster ["Nombre_Zona"] = df_house_cluster["Zona"].map(n_zona) 
df_house_cluster['color_numeric'] = df_house_cluster['Zona'].astype(int) # Tiene que ser entero como en el anterior caso



fig = px.scatter_mapbox(
    df_house_cluster,
    lat = "Lat",
    lon = "Lon",
    color = "Nombre_Zona",
    color_discrete_map = {
        "Centro": "red",
        "Este": "blue",
        "Sureste": "green",
        "Oeste": "purple",
        "Noreste": "orange"
    },
    custom_data=["Price", "Nombre_Zona"],  #  Datos para el hover
    zoom = 10.5,
    height = 600,
 )

fig.update_traces(
    marker=dict(size=6),
    hovertemplate=(
        "💵 <b>Precio:</b> %{customdata[0]:.2f}€<br>"  # Emoji + precio
        "🗺️ <b>Zona:</b> %{customdata[1]}<br>"         # Emoji + zona
        "<extra></extra>"  #  Esto elimina la info extra de Plotly que no queremos que aparezca
    )
)


fig.add_trace(go.Scattermapbox(
    lat = center_cluster["Lat"],
    lon = center_cluster["Lon"],
    mode = "markers",
    marker = dict(
        size = 11,
        color = "black",
        symbol = "circle"
    ),
    hoverinfo = "text",
    text = [f"🌍 {n_zona[i]}" for i in range(len(center_cluster))],
    name = "Centro Zona"
))
  
center_coord = {"lat": center_coords[0] - 0.015, "lon": center_coords[1]}

fig.update_layout(
    mapbox_style = "open-street-map",
    mapbox = dict(center = center_coord),
    margin = {"r": 0, "t": 25, "l": 0, "b": 0}, # Para espacio en blanco
    legend = dict(
        title_text = "Zona",
        yanchor = "top",
        y = 0.99,
        xanchor = "left",
        x = 0.01,
        bgcolor = 'rgba(255, 255, 255, 0.5)' # Fondo semi transparente
    ),
    title = dict(
        text = "🔍🏠Distribución de propiedades por zona🗺️",
        x = 0.5,
        xanchor = "center",
        font = dict(size = 18),
        y = 0.99,
        yanchor = "top",
        pad = dict(b = 0),
        
    ),
    paper_bgcolor='rgba(255, 255, 255, 0.8)') 
        


fig.show()