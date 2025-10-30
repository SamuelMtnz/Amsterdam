# 🏠 Amsterdam Housing Price Predictor
App Streamlit para explorar y predecir precios de viviendas en Ámsterdam en función de la zona donde se encuentran, los metros cuadrados de la vivienda y el número de habitaciones de la misma.

## 🚀 Características Principales

- **🔍 Exploración Interactiva**: Filtra y visualiza propiedades en mapas interactivos
- **🔮 Predicción de Precios**: Modelos de ML especializados por zona geográfica
- **🗺️ Análisis Geográfico**: Clustering de propiedades por ubicación
- **📊 Visualizaciones Avanzadas**: Gráficos interactivos y mapas con Folium

## 🛠️ Tecnologías Utilizadas

- **Frontend**: Streamlit
- **Backend**: Python, Scikit-Learn, XGBoost
- **ML Models**: Random Forest, Linear Regression, XGBoost
- **Visualización**: Plotly, Folium, Seaborn
- **Geolocalización**: Geopy
- **Anñalisis**: Pandas, NumPy, Scipy


## **Fase 1: EDA y Preprocesamiento** 📈

### Análisis Exploratorio de Datos (EDA)
- Carga y validación de datos
- Análisis de valores nulos y duplicados
- Visualizaciones: distribuciones, outliers, correlaciones
- Tratamiento de outliers
- Feature engineering

## **Fase 2: Modelización** 🤖

### Machine Learning
- Comparación de modelos: Random Forest, XGBoost, Linear Regression, SVR, Ridge
- Optimización con GridSearchCV
- Validación cruzada 
- Análisis de residuos
- Guardado de modelos

## **Fase 3: Clustering Geográfico** 🗺️

### Segmentación por ubicación
- K-Means clustering 
- Modelos especializados por zona
- Análisis de precios por cluster
- Visualizaciones con Folium y Plotly

## **Fase 4: Aplicación Interactiva** 🎯

#### Streamlit App
- Exploración de propiedades con filtros
- Predicción de precios en tiempo real
- Mapas interactivos
- Modelo especializado


## **🎯 Características de la Aplicación**

### **🔍 Exploración Interactiva**
- Filtros por zona, habitaciones y área
- Mapa interactivo con propiedades reales
- Detalles al hacer click en propiedades
- Estadísticas en tiempo real

### **🔮 Predicción de Precios**
- Inputs: Área, habitaciones, zona
- Modelo especializado
- Modelo general para otras zonas
- Visualización de ubicación estimada

### **📊 Análisis por Zonas**
- Precios promedios por zona geográfica
- Gráficos interactivos con Plotly
- Centroides de cada zona

## **🤝 Contribuciones**
### Las contribuciones son bienvenidas.
Ponerse en contacto con el autor

## **📄 Licencia**
Este proyecto está bajo la Licencia MIT.

## **👨‍🎓 Autor**
Samuel Martínez
GitHub: @SamuelMtnz