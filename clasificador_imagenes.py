import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Cargar dataset MNIST
digits = datasets.load_digits()

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)


# Función para seleccionar una imagen
def seleccionar_imagen():
    global panel
    ruta_imagen = filedialog.askopenfilename()

    if ruta_imagen:
        imagen = Image.open(ruta_imagen).convert('L')  # Convertir a escala de grises
        imagen = ImageOps.invert(imagen)  # Invertir colores para parecerse a MNIST
        imagen = imagen.resize((8, 8))  # Redimensionar a 8x8
        imagen_array = np.array(imagen)

        imagen_array = imagen_array / 16.0  # Normalizar valores (MNIST va de 0-16)
        imagen_array = imagen_array.flatten()

        prediccion = model.predict([imagen_array])[0]

        # Mostrar imagen
        imagen = Image.open(ruta_imagen)
        imagen = imagen.resize((150, 150))
        imagen = ImageTk.PhotoImage(imagen)

        if panel is None:
            panel = tk.Label(image=imagen)
            panel.image = imagen
            panel.pack(padx=10, pady=10)
        else:
            panel.configure(image=imagen)
            panel.image = imagen

        # Mostrar predicción
        etiqueta_resultado.config(text=f"Predicción: {prediccion}")


# Crear ventana principal
ventana = tk.Tk()
ventana.title("Clasificador de Imágenes MNIST")

panel = None

btn_seleccionar = tk.Button(ventana, text="Seleccionar Imagen", command=seleccionar_imagen)
btn_seleccionar.pack(pady=20)

etiqueta_resultado = tk.Label(ventana, text="Predicción: ", font=("Helvetica", 16))
etiqueta_resultado.pack(pady=10)

ventana.mainloop()
