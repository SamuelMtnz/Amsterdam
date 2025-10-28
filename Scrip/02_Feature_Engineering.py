# Importamos librerías
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from scipy.stats import skew

# Librerías gráficos
import seaborn as sns 
import matplotlib.pyplot as plt

import joblib
from pathlib import Path

# Importamos DB
df_house = pd.read_csv('Amsterdam\data\HousingPrices-Amsterdam-August-2021.csv')
    
# Guardamos copia para modificaciones
df_house_copy = df_house.copy()


#--------------------------------------------
#         Valores nulos o fatantes
#--------------------------------------------

# Eliminamos las columnas irrelevantes
df_house_copy = df_house_copy.drop(['Unnamed: 0', 'Address', 'Zip'], axis = 1) #axis = 1 para columnas
print('Columnas restantes:',df_house_copy.columns.tolist())

# Valores nulos
null_values = df_house_copy.isnull().sum()
print('Valores nulos por columna:\n"',null_values)

null_value = df_house_copy[df_house_copy.isnull().any(axis = 1)] #axis = 1 para ver las filas 
print('Filas con valores nulos:\n', null_value)

# Sustituir valores nulos
room12 = df_house_copy[df_house_copy['Room'] == 12]
print('\nMedia de casas con habitaciones = 12 \n', room12.describe())

mean_price = df_house_copy['Price'].mean()

df_house_copy.loc[
    (df_house_copy['Room'] == 12) & (df_house_copy['Price'].isnull()), #Condiciones
    'Price' # Variable a modificar
] = mean_price

room3 = df_house_copy[
    (df_house_copy['Room'] == 3) &
    (df_house_copy['Area'] >= 81) &
    (df_house_copy['Area'] <= 147)
]
print('\nMedia de casas con habitaciones = 3 \n', room3.describe())

f_room3 = (df_house_copy['Room'] == 3) & (df_house_copy['Area'].between(81, 147)) # Realizamos una máscara para poder filtrar posteriormente
mean_room3 = df_house_copy[f_room3]['Price'].mean()
df_house_copy.loc[
    (df_house_copy['Price'].isnull()) & f_room3,
    'Price'
] = mean_room3

print('\n🔍 Nulos en room = 12: ', df_house_copy[df_house_copy['Room'] == 12]['Price'].isnull().sum()) 
print('\n🔍 Nulos en room = 3: ', df_house_copy[f_room3]['Price'].isnull().sum())

# Skewness / Asimetría
skewness = skew(df_house_copy['Price'])
print(f'\nCoeficiente de asimetría: {skewness}')

numeric_cols = df_house_copy.select_dtypes(include = ['float64', 'int64']).columns
for col in numeric_cols:
    print(f'\nSkewness de {col}: {skew(df_house_copy[col])}')

    
# Outliers
def dec_outliers (df_house_copy, column):
    Q1 = df_house_copy[column].quantile(0.25)
    Q3 = df_house_copy[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df_house_copy[(df_house_copy[column] < lower_bound) | (df_house_copy[column] > upper_bound)]

outliers = {col: dec_outliers(df_house_copy, col) for col in df_house_copy.select_dtypes(include = ['float64', 'int64']).columns}

var3 = ['Price', 'Area', 'Room']

plt.figure(figsize = (12, 6))
for i, var in enumerate(var3, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x = df_house_copy[var])
    plt.title(var)
plt.tight_layout()
plt.show()

out_col = {}
for col in var3:
    out_col[col] = dec_outliers(df_house_copy, col)
    print(f'\nOutliers e {col}: {len(out_col[col])} registros')
    
plt.figure(figsize = (12, 6))
for i, var in enumerate(var3, 1):
    plt.subplot(2, 2, i)
    plt.boxplot(x = df_house_copy[var])
    plt.title(var)
plt.tight_layout()
plt.show()

for col in var3:
    print(f"\n=== Análisis de {col} ===")
    print("Estadísticas descriptivas:")
    print(df_house_copy[col].describe())

    # Calcular outliers usando tu función
    outliers = dec_outliers(df_house_copy, col)
    print(f"\nNúmero de outliers en {col}: {len(outliers)}")
    print("Estadísticas de los outliers:")
    print(outliers[col].describe())
    
# Crear una columna binaria (1: outlier, 0: normal) para cada característica
for col in var3:
    df_house_copy[f'outlier_{col}'] = 0  # Inicializar en 0
    outliers = dec_outliers(df_house_copy, col)
    df_house_copy.loc[outliers.index, f'outlier_{col}'] = 1  # Marcar outliers

# Filtrar outliers en ambas columnas
outliers_price_area = df_house_copy[(df_house_copy['outlier_Price'] == 1) & (df_house_copy['outlier_Area'] == 1)]

# Gráfico de dispersión
plt.figure(figsize=(10, 6))
sns.scatterplot(x = 'Area', y = 'Price', data = df_house_copy, color = 'blue', label = 'Datos Normales')
sns.scatterplot(x = 'Area', y = 'Price', data = outliers_price_area, color = 'red', label = 'Outliers en Ambos')
plt.title('Relación entre Area y Price con Outliers Destacados')
plt.legend()
plt.show()

# Representación en mapa
import folium
from folium.plugins import HeatMap
import numpy as np


center_coords = (52.377956, 4.897070) # Lon y Lat de Amsterdam

mapa_out = folium.Map(location = center_coords, zoom_start = 12)

colores_out = {
    'Price' : 'red',
    'Area' : 'blue',
    'Room' : 'green'
}

for columna in out_col.keys():
    outliers = out_col[columna]
    for idx, row in outliers.iterrows():
        folium.CircleMarker(
            location = [row['Lat'], row['Lon']],
            radius = 2,
            color = colores_out[columna],
            fill = True,
            fill_color = colores_out[columna],
            fill_opacity = 0.7,
            popup = f"{columna}: {row[columna]}"
        ).add_to(mapa_out)

#Guardamos mapa
mapa_out.save('Amsterdam/Graphs/mapa_outliers.html')
import IPython 
IPython.display.HTML(mapa_out._repr_html_())



# WINSORIZACIÓN

# Se sustituyen los valores por percentil más alto y más bajo

var3 = df_house_copy[['Area', 'Price', 'Room']]

for col in var3:
  p_low = df_house_copy[col].quantile(0.01)
  p_high = df_house_copy[col].quantile(0.99)

  if p_high <= p_low:
      p_high = p_low + 1e-6


  df_house_copy[col + '_winsor'] = df_house_copy[col].clip(lower = p_low, upper = p_high)


print(df_house_copy.describe().T)





# Eliminamos Outliers
for col in outliers:
  df_house_copy = df_house_copy[~df_house_copy.index.isin(outliers[col].index)]

# Añadimos varibles, no relaciónas con la variable a predecir para evitar overfiting

df_house_copy['Room_Area'] = df_house_copy['Area'] / df_house_copy['Room']


df_house_final = df_house_copy[['Area', 'Price', 'Room', 'Room_Area', 'Lon', 'Lat']].copy()
print(f'\nVariables en el DF: {df_house_final.columns.tolist()}')

# Guardar DB
df_house_final.to_csv('Amsterdam\data\datos_procesados.csv', index=False)