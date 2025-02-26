import json
import matplotlib.pyplot as plt

# Cargar el history desde el archivo JSON
with open("/home/miguelbareno/Desktop/bootcamp_artificial_intelligence/classification_proyect/code/training_history.json", "r") as f:
    history = json.load(f)

# Graficar Precisión
plt.plot(history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history['val_accuracy'], label='Precisión en validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.title('Precisión del modelo')
plt.show()

# Graficar Pérdida
plt.plot(history['loss'], label='Pérdida en entrenamiento')
plt.plot(history['val_loss'], label='Pérdida en validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida del modelo')
plt.show()
