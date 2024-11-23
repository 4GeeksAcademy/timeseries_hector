# Importación de librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")

# Paso 1: Cargar datos
def load_data(file_path):
    """
    Carga los datos del archivo CSV y selecciona las columnas relevantes.
    """
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data = data[['water_amount']].dropna()  # Seleccionamos la columna principal
    print("Vista previa de los datos:")
    print(data.head())
    return data

# Paso 2: Exploración de datos
def explore_data(data):
    """
    Exploración inicial de los datos.
    """
    print("\nInformación del dataset:")
    print(data.info())
    print("\nDescripción estadística:")
    print(data.describe())
    
    # Visualización de la serie temporal
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Datos históricos')
    plt.title('Evolución de la cantidad de agua')
    plt.xlabel('Fecha')
    plt.ylabel('Cantidad de agua')
    plt.legend()
    plt.show()

# Paso 3: División de datos en entrenamiento y prueba
def split_data(data, train_size=0.8):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    split_index = int(len(data) * train_size)
    train = data[:split_index]
    test = data[split_index:]
    print(f"Datos de entrenamiento: {len(train)}, Datos de prueba: {len(test)}")
    return train, test

# Paso 4: Modelado con opciones para datos limitados
def train_model(train):
    """
    Entrena un modelo de Holt-Winters ajustado para datos limitados.
    """
    try:
        # Intentar con estacionalidad
        model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12, trend='add').fit()
    except ValueError as e:
        print("Advertencia:", e)
        print("Cambiando a modelo sin estacionalidad...")
        # Si falla, desactiva la estacionalidad
        model = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
    print("Modelo entrenado con éxito.")
    return model

# Paso 5: Evaluación del modelo
def evaluate_model(model, train, test):
    """
    Evalúa el modelo y realiza predicciones.
    """
    # Predicción en datos de entrenamiento
    train_pred = model.fittedvalues
    
    # Predicción en datos de prueba
    test_pred = model.forecast(len(test))
    
    # Métricas de error
    train_rmse = np.sqrt(mean_squared_error(train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(test, test_pred))
    
    print(f"RMSE en datos de entrenamiento: {train_rmse:.2f}")
    print(f"RMSE en datos de prueba: {test_rmse:.2f}")
    
    # Visualización de resultados
    plt.figure(figsize=(12, 6))
    plt.plot(train, label='Entrenamiento')
    plt.plot(test, label='Prueba')
    plt.plot(test_pred, label='Predicción', linestyle='--')
    plt.title('Resultado del modelo Holt-Winters')
    plt.legend()
    plt.show()

    return test_pred

# Paso 7: Modelo naive como fallback
def naive_forecast(train, test):
    """
    Pronóstico naive: la predicción es igual al último valor observado.
    """
    last_value = train.iloc[-1].values[0]
    predictions = pd.Series([last_value] * len(test), index=test.index)  # Aseguramos que las predicciones tengan el mismo índice
    print("Usando modelo naive debido a datos limitados.")
    return predictions


# Paso 6: Guardar predicciones en un archivo
def save_predictions(test, predictions, output_file='predictions.csv'):
    """
    Guarda las predicciones en un archivo CSV.
    """
    result = pd.DataFrame({
        'Actual': test['water_amount'],
        'Predicted': predictions
    })
    
    result.to_csv(output_file)
    print(f"Predicciones guardadas en {output_file}")



# Flujo principal
if __name__ == "__main__":
    # Cambia 'data.csv' por la ruta al archivo proporcionado
    file_path = '/workspace/timeseries_hector/data/interim/data.csv'
    
    # Cargar y explorar los datos
    data = load_data(file_path)
    explore_data(data)
    
    # División de datos
    train, test = split_data(data)
    
    # Verificar si hay suficientes datos para modelos complejos
    if len(train) < 12:  # Si hay menos de 12 puntos, usa el modelo naive
        predictions = naive_forecast(train, test)
        test['Predicted'] = predictions
        save_predictions(test, np.array(predictions), output_file='predictions_naive.csv')
    else:
        # Entrenamiento del modelo
        model = train_model(train)
        
        # Evaluación del modelo
        test_predictions = evaluate_model(model, train, test)
        
        # Guardar predicciones
        save_predictions(test, test_predictions)
