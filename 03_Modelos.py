#--------------------------------------------
#               MODELIZACIÓN
#--------------------------------------------

#Importamos librerías
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from scipy.stats import skew

#Librerías gráficos
import seaborn as sns 
import matplotlib.pyplot as plt

#Librerías modelos
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.svm import SVR

import joblib
from scipy.stats import randint

# Cross Validation y Pipelines
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline


# Cargar DB
import pandas as pd
df_house_final = pd.read_csv('datos_procesados.csv')



X = df_house_final[['Area', 'Room', 'Room_Area', 'Lon', 'Lat']]
y = df_house_final['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)

print(f"📊 Tamaño de los datos:")
print(f"- 🧑‍🎓 Entrenamiento: {X_train.shape[0]} filas")
print(f"- 🧑‍🏫 Prueba: {X_test.shape[0]} filas")


modelos = {
    'Regrasión Lineal' : make_pipeline(StandardScaler(), LinearRegression()),
    'Random Forest' :  RandomForestRegressor(random_state = 42),
    'XGBoost' : make_pipeline(StandardScaler(),XGBRegressor(random_state = 42)),
    'SVR' : SVR(),
    'Ridge' : make_pipeline(StandardScaler(), Ridge())
}

resultados = {}

for nombre ,modelo in modelos.items():
    # Entrenamiento
    modelo.fit(X_train, y_train)
    
    # Predicción
    y_pred = modelo.predict(X_test)
    
    # Mátricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross validation
    cv_scores = cross_val_score(modelo, X_train, y_train, cv = 5, scoring = 'r2')
    avg_r2_cv = np.mean(cv_scores)    
    
    resultados[nombre] = {"MSE": mse, "R²": r2, "MAE": mae, 'R² CV': avg_r2_cv}

print("\n📈 Resultados de los modelos:")
for modelo, metrics in resultados.items():
    print(f"\n🔹 {modelo}:")
    print(f"- MSE: {metrics['MSE']:.2f}")
    print(f"- R²: {metrics['R²']:.2f}")
    print(f"- MAE: {metrics['MAE']:.2f}")
    print(f"- R² CV: {metrics['R² CV']:.2f}")
    
mejor_modelo = max(resultados, key=lambda x: resultados[x]['R²'])
print(f"\n🏆 Mejor modelo: {mejor_modelo} (R² = {resultados[mejor_modelo]['R²']:.3f})")


# --------------------------------------------
# Optimizar hiperparámetros GridSearch
# --------------------------------------------
param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ['sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator = rf, 
    param_grid = param_grid,
    cv = 5,                     # 5-fold cross validation
    scoring = 'r2',             # Optimizar para R²
    verbose = 1,                # Mostrar progreso
    n_jobs = -1                 # Usar todos los núcleos del CPU
)

grid_search.fit(X_train, y_train)

# Mejores parámetros
print(f"\n🎯 Mejores parámetros: {grid_search.best_params_}")

# Evaluar modelo optimizado
mejor_rf = grid_search.best_estimator_
y_pred_rf = mejor_rf.predict(X_test)
print(f"R² optimizado: {r2_score(y_test, y_pred_rf):.2f}")

# Analizar Residuos
residuals = y_test - y_pred_rf
plt.scatter(y_pred, residuals)
plt.axhline(y = 0, color = 'r', linestyle = '-')
plt.title("Análisis de residuos")
plt.xlabel("Predicciones")
plt.ylabel("Residuos")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde = True, bins = 30)
plt.title("Distribución de Residuos")
plt.xlabel("Residuos")
plt.show()



# Importar librerías para directorios
import joblib
from pathlib import Path

# Crear directorio para modelos
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Guardar el mejor modelo general
joblib.dump(mejor_rf, MODELS_DIR / "modelo_general.pkl")


# Guardar información de características
caracteristicas = list(X.columns)
joblib.dump(caracteristicas, MODELS_DIR / "caracteristicas.pkl")

print("✅ Modelos guardados:")
print(f"- Modelo general: models/modelo_general.pkl")
