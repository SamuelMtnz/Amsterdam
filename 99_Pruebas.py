import pandas as pd
import numpy as np

df = pd.read_csv("C:\\Users\\Samu\\Documents\\Nodd3r\\Amsterdam\\datos_cluster.csv")

print(df.columns.tolist())

# 2. Obtener las 5 áreas más pequeñas
print("\n5 áreas más pequeñas:")
area = df['Area'].sort_values(ascending= False).head(5)
print(area)

hab = df['Room'].sort_values(ascending = False).head(5)
print(hab)

f1= df[
    (df["Zona"] == 1) &
    (df["Room_Area"].between(10,50))
]

print(f'{f1.sort_values(by = "Room_Area").head(3)}') #Para ordenar se necesita by =
print(f'{f1[['Area', 'Price', 'Room', 'Room_Area', 'Zona']].sort_values(by = "Room_Area").head(3)}') #Para mostrar las variables que queremos