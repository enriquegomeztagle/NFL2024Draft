import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Cargar los datos para los QBs
df = pd.read_csv('data.csv')

# Reemplazar los guiones '-' con NaN para manejar esos valores
df.replace('-', pd.NA, inplace=True)

# Se convierten a numéricas las columnas 'PYD', 'TD', 'INT', 'CMP%', 'Y/A', 'RUSH' y 'RATE'
# para evitar errores durante la conversión
df['PYD'] = pd.to_numeric(df['PYD'], errors='coerce')
df['TD'] = pd.to_numeric(df['TD'], errors='coerce')
df['INT'] = pd.to_numeric(df['INT'], errors='coerce')
df['Y/A'] = pd.to_numeric(df['Y/A'], errors='coerce')
df['RUSH'] = pd.to_numeric(df['RUSH'], errors='coerce')

# Convertir el porcentaje a un valor decimal
df['CMP%'] = pd.to_numeric(df['CMP%'].str.rstrip('%'), errors='coerce') / 100

# Convertir la columna 'RATE' a numérica y manejar los valores no numéricos
df['RATE'] = pd.to_numeric(df['RATE'], errors='coerce')

# Imputar valores faltantes
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df[['PYD', 'TD', 'INT', 'CMP%', 'Y/A', 'RUSH', 'RATE']]), 
                          columns=['PYD', 'TD', 'INT', 'CMP%', 'Y/A', 'RUSH', 'RATE'])

# Seleccionar las características (X) y la variable objetivo (y)
X = df_imputed[['PYD', 'TD', 'CMP%', 'Y/A', 'RUSH']]
y = df_imputed['RATE']

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Se usa el 30% de los datos para pruebas
# Se usa random_state para asegurar reproducibilidad

# Regresión lineal
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
gb_y_pred = gb_model.predict(X_test)

# Validación cruzada con Regresión Lineal
lr_cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='neg_mean_squared_error')
lr_cv_rmse_scores = np.sqrt(-lr_cv_scores)

# Validación cruzada con Random Forest
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')
rf_cv_rmse_scores = np.sqrt(-rf_cv_scores)

# Validación cruzada con Gradient Boosting
gb_cv_scores = cross_val_score(gb_model, X, y, cv=5, scoring='neg_mean_squared_error')
gb_cv_rmse_scores = np.sqrt(-gb_cv_scores)

# Evaluar los modelos
lr_mse = mean_squared_error(y_test, lr_y_pred)
rf_mse = mean_squared_error(y_test, rf_y_pred)
gb_mse = mean_squared_error(y_test, gb_y_pred)

lr_mae = mean_absolute_error(y_test, lr_y_pred)
rf_mae = mean_absolute_error(y_test, rf_y_pred)
gb_mae = mean_absolute_error(y_test, gb_y_pred)

lr_r2 = r2_score(y_test, lr_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)
gb_r2 = r2_score(y_test, gb_y_pred)

# Mostrar los resultados
print(f"Linear Regression MSE: {lr_mse}")
print(f"Linear Regression Cross-Validation RMSE (mean): {np.mean(lr_cv_rmse_scores)}")
print(f"Linear Regression MAE: {lr_mae}")
print(f"Linear Regression R²: {lr_r2}")

print(f"Random Forest MSE: {rf_mse}")
print(f"Gradient Boosting MSE: {gb_mse}")
print(f"Random Forest Cross-Validation RMSE (mean): {np.mean(rf_cv_rmse_scores)}")
print(f"Gradient Boosting Cross-Validation RMSE (mean): {np.mean(gb_cv_rmse_scores)}")
print(f"Random Forest MAE: {rf_mae}")
print(f"Gradient Boosting MAE: {gb_mae}")
print(f"Random Forest R²: {rf_r2}") 
print(f"Gradient Boosting R²: {gb_r2}")
