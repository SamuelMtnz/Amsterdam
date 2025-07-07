#Importamos las librerías necesarias para el trabajo de EDA

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import os
from pathlib import Path



# Cargamos archivo
try:
    # # Pedimos la ruta del archivo al usuario
    # ruta_archivo = input("📁 Ingresa la ruta completa del archivo CSV: ").strip()

    # # Normalizamos la ruta para Windows
    # ruta_archivo = os.path.normpath(ruta_archivo)
    
    # # Comprobación carga archivo. Usamos raise para controlar el flujo y tener un error claro
    # if not ruta_archivo:
    #     raise FileNotFoundError('⚠️ No se ha ingresado ninguna ruta')
    
    # # Obtenemos solo el nombre del archivo para mostrar
    # nombre_archivo = os.path.basename(ruta_archivo)
    # print(f'\n 🔍 Archivo seleccionado: {nombre_archivo}')
    
    # # Comprobación tipo archivo usando os.path.splitext
    # if os.path.splitext(ruta_archivo)[1].lower() != '.csv':
    #     print(f"Extensión detectada: {os.path.splitext(ruta_archivo)[1]}")
    #     raise ValueError('❌ El archivo no es un CSV')
    
    # # Verificar si el archivo existe
    # if not os.path.exists(ruta_archivo):
    #     raise FileNotFoundError(f"⚠️ Archivo no encontrado: {ruta_archivo}")
    
    # # Carga datos en un DF y verificación - CORRECCIÓN: pd.read_csv()
    # df_house = pd.read_csv(ruta_archivo)
    
    df_house = pd.read_csv('C:\\Users\\Samu\\Desktop\\Nodd3r\\Proyectos\\1_Amsterdam\\HousingPrices-Amsterdam-August-2021.csv')
    
    print('\n ✅ Datos cargados correctamente \n Vista previa: ')
    print(df_house.head())
    
    print('\n 📊 Resúmen estadístico básico: ')
    print(df_house.describe())
    
    # Manejo Errores
    
except FileNotFoundError as e:
    print(f'\n ❌ Error: {str(e)}')
except ValueError as e:
    print(f'\n ❌ Error: {str(e)}')
except pd.errors.EmptyDataError:
    print('\n ❌ El archivo está vacío.')
except pd.errors.ParserError:
    print('\n ❌ Error al leer archivo, revisar formato.')
except Exception as e:
    print(f'\n ❌ Error inesperado: {str(e)}')
    

# Información variables y duplicados

df_house.info()

rows_duplicated = df_house.duplicated()
n_rows_duplicated = df_house.duplicated().sum()
print(f'\n 🔍 Columnas duplicadas: {n_rows_duplicated}')

# Descripción gráfica

n_col = df_house.select_dtypes(include = ['float64', 'int64']).columns

plt.figure(figsize = (12, 6))

for i, col in enumerate(n_col, 1):
    plt.subplot(3, 2, i)
    sns.histplot(df_house[col], kde = True)
    plt.title(f'Distribución de: {col}')
    plt.xlabel(col)
    plt.ylabel(f'Frecuencia')
    
plt.tight_layout()
plt.show()

# Outliers, creamos un a variable que recoja las dos características que queremos analizar


plt.figure(figsize = (14, 10))

for i, col in enumerate(['Area', 'Room'], 1):
    plt.subplot(2, 1, i)
    sns.boxplot(x = col, y = "Price", data = df_house)
    plt.title(f"Precio vs {col}")
    plt.xlabel(col)
    plt.ylabel("Precio")

plt.tight_layout()
plt.show()

# Distribución de las casas por precio, área y número de habitaciones

plt.figure(figsize = (12,6))

sns.scatterplot(x = "Price",
                y = "Area",
                data = df_house,
                hue = "Room",
                palette = "viridis")

plt.title("Price vs Area for Rooms")
plt.xlabel("Price")
plt.ylabel("Area")

plt.legend(title = "Room",
           loc = "upper right")

plt.show()

# Correlaciones

plt.figure(figsize = (12, 6))

corr_matrix = df_house.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr_matrix, annot = True, cmap = "coolwarm")
plt.title("Mapa de calor de Correlaciones")

plt.show()

# Pairplot e ydata

sns.pairplot(df_house)

plt.show()



# from pandas_profiling  import ProfileReport

# profile_report = ProfileReport(df_house, title = "Reporte EDA", explorative = True)
# profile_report.to_notebook_iframe()