import cv2
import matplotlib.pyplot as plt
import os
import time

def visualize_image(image_path, x=50, y=50, width=100, height=100):
    """
    Carga y muestra una imagen con un recuadro dibujado en la posición indicada.
    """
    try:
        x = round(float(x))
        y = round(float(y))
        width = round(float(width))
        height = round(float(height))
        if not os.path.isfile(image_path):
            print("El archivo no existe.")
            return
        
        # Cargar la imagen
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"cargar la imagen {len(image)}, x: {x}, y: {y}, width: {width}, height:{height}")
        
        # Dibujar el recuadro
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
        
        # Mostrar la imagen
        plt.imshow(image)
        plt.axis("off")
        plt.show(block=True)
        
        # Pausar 5 segundos y cerrar la imagen
        # time.sleep(5)
        # plt.close()
    except Exception as e:
        print(f"Error visualizando la imagen: {e}")
        pass

# Ejemplo de uso
# visualize_image("ruta/a/la/imagen.jpg", x=30, y=30, width=150, height=150)
