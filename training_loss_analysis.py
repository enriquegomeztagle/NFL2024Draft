import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv('data.csv')

# Preprocesamiento de datos
df.replace('-', pd.NA, inplace=True)  # Reemplazar '-' por NaN (valores faltantes)
for col in ['PYD', 'TD', 'INT', 'Y/A', 'RUSH', 'RATE']:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convertir a numérico

df['CMP%'] = pd.to_numeric(df['CMP%'].str.rstrip('%'), errors='coerce') / 100  # Convertir porcentaje a decimal

# Imputación de valores faltantes
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df[['PYD', 'TD', 'INT', 'CMP%', 'Y/A', 'RUSH', 'RATE']]), 
                          columns=['PYD', 'TD', 'INT', 'CMP%', 'Y/A', 'RUSH', 'RATE'])

# Definir características (X) y variable objetivo (y)
X = df_imputed[['PYD', 'TD', 'CMP%', 'Y/A', 'RUSH']]
y = df_imputed['RATE']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Sección de Modelos y Entrenamiento

# Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
gb_train_loss = gb_model.train_score_  # Obtener pérdida durante el entrenamiento

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, oob_score=True, warm_start=True, random_state=42)
oob_errors = []
for n_trees in range(1, 101):
    rf_model.set_params(n_estimators=n_trees)
    rf_model.fit(X_train, y_train)
    oob_error = 1 - rf_model.oob_score_  # Calcular error OOB
    oob_errors.append(oob_error)

# Sección de Visualización

# Gráfica de pérdida de Gradient Boosting
plt.figure(figsize=(10, 6))
plt.plot(gb_train_loss, label='Training Loss')
plt.xlabel('Boosting Iterations')
plt.ylabel('Loss')
plt.title('Gradient Boosting Training Loss')
plt.legend()
plt.show()

# Gráfica de error OOB de Random Forest
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), oob_errors, label='OOB Error')
plt.xlabel('Number of Trees')
plt.ylabel('OOB Error')
plt.title('OOB Error Across Number of Trees in Random Forest')
plt.legend()
plt.show()
