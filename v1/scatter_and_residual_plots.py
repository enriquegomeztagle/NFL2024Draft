import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
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

# Regresión Lineal
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)

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


# Crear una figura con 2 filas y 3 columnas de subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Gráficos de dispersión y residuos para cada modelo
models = {
    'Linear Regression': lr_model,
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model
}

for i, (model_name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Scatter plot (Valores reales vs. predichos)
    row = i // 3 
    col = i % 3 
    axes[row, col].scatter(y_test, y_pred, color='blue', label='Predicted')
    axes[row, col].scatter(y_test, y_test, color='red', label='Real')
    axes[row, col].set_xlabel('Real Values')
    axes[row, col].set_ylabel('Predicted Values')
    axes[row, col].set_title(f'Scatter Plot: {model_name}')
    axes[row, col].legend()

    # Residual plot (Residuos vs. valores predichos)
    row = 1
    col = i % 3
    axes[row, col].scatter(y_pred, residuals)
    axes[row, col].set_xlabel('Predicted Values')
    axes[row, col].set_ylabel('Residuals')
    axes[row, col].set_title(f'Residual Plot: {model_name}')
    axes[row, col].axhline(y=0, color='r', linestyle='--')

plt.tight_layout()
plt.show()
