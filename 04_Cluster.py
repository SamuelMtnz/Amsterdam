from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

# Cargar DB

df_house_final = pd.read_csv('datos_procesados.csv')

df_house_cluster = df_house_final.copy()

# --------------------------------------------
#                 Clusters
# --------------------------------------------

# Seleccionar coordenadas
coordenadas = df_house_cluster[["Lon", "Lat"]]

# Crear clusters
kmeans = KMeans(n_clusters = 5, random_state = 42)
df_house_cluster["Zona"] = kmeans.fit_predict(coordenadas)

plt.figure(figsize = (12, 6))

# Colores para cada zona
colores = ["red", "blue", "green", "purple", "orange"]

# Scatterplot por zona
for zona in range(5):
    datos_zona = df_house_cluster[df_house_cluster["Zona"] == zona]
    plt.scatter(
        datos_zona["Lon"],
        datos_zona["Lat"],
        color = colores[zona],
        label = f"Zona {zona}",
        alpha = 0.6
    )

# Añadir detalles
plt.title("Distribución de Casas por Zona Geográfica", fontsize = 14)
plt.xlabel("Longitud (Lon)", fontsize = 12)
plt.ylabel("Latitud (Lat)", fontsize = 12)
plt.legend()
plt.grid(True, linestyle = "--", alpha = 0.5)
plt.savefig("zonas_geograficas.png")
plt.show()

precio_por_zona = df_house_cluster.groupby("Zona")["Price"].mean().reset_index()

plt.figure(figsize = (10, 6))
sns.barplot(x = "Zona", y = "Price", data = precio_por_zona, palette = colores)
plt.title("Precio Promedio por Zona Geográfica")
plt.xlabel("Zona")
plt.ylabel("Precio Medio (€)")
plt.savefig("precio_por_zona.png")
plt.show()

print(df_house_cluster["Zona"].value_counts())

models_zone = {}


for zone in df_house_cluster["Zona"].unique():
    zone_data = df_house_cluster[df_house_cluster["Zona"] == zone]
# --------------------------------------------
# Definir variables (X) y target (y)
# --------------------------------------------

    X_cluster = zone_data[["Area", "Room", "Lat", "Lon", "Room_Area"]]
    y_cluster = zone_data["Price"]

    X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(X_cluster, y_cluster, test_size = 0.2, random_state = 42)

    print(f"\n📊 Tamaño de los datos:")
    print(f"- 🧑‍🎓 Entrenamiento: {X_train_cluster.shape[0]} filas")
    print(f"- 🧑‍🏫 Prueba: {X_test_cluster.shape[0]} filas")


  # --------------------------------------------
  # Escalar datos
  # --------------------------------------------

    scaler_cluster = StandardScaler()
    X_train_scaled_cluster = scaler_cluster.fit_transform(X_train_cluster)
    X_test_scaled_cluster = scaler_cluster.transform(X_test_cluster)


  # --------------------------------------------
  #                Entrenar modelos
  # --------------------------------------------

    modelos_cluster = {
    "XGBoost": XGBRegressor(objective = 'reg:squarederror', random_state = 42),
    "Regresión Lineal": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state = 42),
    "SVR": SVR(),
    "Ridge": Ridge()
  }

    resultados_cluster = {}

    for nombre_cluster, modelo_cluster in modelos_cluster.items():
      # Entrenar modelo
       modelo_cluster.fit(X_train_scaled_cluster, y_train_cluster)

      # Predecir
       y_pred_cluster = modelo_cluster.predict(X_test_scaled_cluster)

      # Calcular métricas
       mse = mean_squared_error(y_test_cluster, y_pred_cluster)
       r2 = r2_score(y_test_cluster, y_pred_cluster)
       mae = mean_absolute_error(y_test_cluster, y_pred_cluster)

       resultados_cluster[nombre_cluster] = {'MSE': mse, 'MAE': mae, 'R2': r2}

# Mostrar resultados
    print(f"\n📈 Resultados de los modelos {zone}:")
    for modelo_cluster, metrics_cluster in resultados_cluster.items():
      print(f"\n🔹 {modelo_cluster}:")
      print(f"- MSE: {metrics_cluster['MSE']:.2f}")
      print(f"- R²: {metrics_cluster['R2']:.2%}")
      print(f"- MAE: {metrics_cluster['MAE']:.2f}")

# Guardamos DB
df_house_cluster.to_csv('datos_cluster.csv', index=False)



import joblib
from pathlib import Path

# Crear directorio para modelos
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Guardar modelo KMeans
joblib.dump(kmeans, MODELS_DIR / "kmeans_model.pkl")


# Filtrar datos solo para Zona 0
zona_0_data = df_house_cluster[df_house_cluster["Zona"] == 0]

# Verificar que haya datos para la Zona 0
if not zona_0_data.empty:
    # Preparar datos para Zona 0
    X_cluster0 = zona_0_data[["Area", "Room", "Lat", "Lon", "Room_Area"]]
    y_cluster0 = zona_0_data["Price"]
    
    # Entrenar modelo especial para Zona 0
    modelo_cluster0 = RandomForestRegressor(
        n_estimators = 300,
        max_depth = 20,
        min_samples_split = 5,
        random_state = 42
    )
    modelo_cluster0.fit(X_cluster0, y_cluster0)
    
    # Guardar modelo
    joblib.dump(modelo_cluster0, MODELS_DIR / "modelo_zona_0_especial.pkl")
    print("✅ Modelo especial para Zona 0 guardado")
else:
    print("⚠️ Advertencia: No hay datos para la Zona 0")

# Guardar datos con clusters para mapas
df_house_cluster.to_csv('datos_cluster.csv', index = False)
print("✅ Datos con clusters guardados")