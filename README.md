# Pasos para el Entrenamiento de un Modelo en TensorFlow
## Instalamos los módulos necesarios
pip install tensorflow

pip install numpy (Este módulo permite realizar cálculos con Python, por lo general viene instalado por defecto)

pip install matplotlib 
## Importar TensorFlow y NumPy
import tensorflow as tf
import numpy as np
## Definir los datos de entrada
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit= np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
###### Le damos ejemplos a la red neuronal de la conversión de Celsius a Fahrenheit
## Crear una Capa en el Modelo
layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])
###### Creamos una capa en el modelo con Dense lo que conectara una a varias unidades
###### pero en este caso solo usamos una de entrada y una de salida asi que la dejamos
###### en 1 y usamos un modelo secuencial.
## Compilar el Modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Entrenar el modelo
print("Starting training...")
history = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Trained model")

## Visualizar la pérdida durante el entrenamiento
import matplotlib.pyplot as plt
plt.xlabel("# Period")
plt.ylabel("Magnitude of loss")
plt.plot(history.history["loss"])
plt.show()

## Realizar una predicción
print("Let's make a prediction!")
result = model.predict([100.0])
print("The result is " + str(result) + " Fahrenheit")
###### En este caso yo use 100 como entrada a Celsius

## Imprimir las variables internas del modelo (pesos y sesgos de la capa)
print("Internal model variables")
print(layer.get_weights())
