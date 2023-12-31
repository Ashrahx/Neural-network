import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Definir los datos de entrada
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Crear una capa en el modelo
layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])

# Compilar el modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Entrenar el modelo
print("Starting training...")
history = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Trained model")

# Visualizar la pérdida durante el entrenamiento
plt.xlabel("# Period")
plt.ylabel("Magnitude of loss")
plt.plot(history.history["loss"])
plt.show()

# Realizar una predicción
print("Let's make a prediction!")
result = model.predict([100.0])
print("The result is " + str(result) + " Fahrenheit")

# Imprimir las variables internas del modelo
print("Internal model variables")
print(layer.get_weights())
