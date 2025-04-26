import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# 🛠️ Entrenar el modelo si no existe
def entrenar_modelo():
    if not os.path.exists('modelo_entrenado.pkl'):
        print("Entrenando modelo, espera un momento...")
        digits = datasets.load_digits()
        X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
        modelo = LogisticRegression(max_iter=10000)
        modelo.fit(X_train, y_train)
        joblib.dump(modelo, 'modelo_entrenado.pkl')
        print("✅ Modelo entrenado y guardado.")
    else:
        print("Modelo ya entrenado.")

# 🔥 Preparar la imagen cargada
def preparar_imagen(ruta_imagen):
    imagen = Image.open(ruta_imagen).convert('L')  # Escala de grises
    imagen = imagen.resize((8, 8))                 # Redimensionar a 8x8
    imagen = np.array(imagen)
    imagen = 16 - (imagen / 16)                    # Invertir color para parecerse a digits
    imagen = imagen.flatten()
    return imagen

# 🚀 Cargar imagen y predecir
def seleccionar_imagen():
    ruta = filedialog.askopenfilename()
    if ruta:
        imagen = Image.open(ruta)
        imagen_resized = imagen.resize((200, 200))  # Mostrar imagen en tamaño más grande
        imagen_tk = ImageTk.PhotoImage(imagen_resized)

        panel.config(image=imagen_tk)
        panel.image = imagen_tk  # Importante para no perder referencia

        # Preparar la imagen para el modelo
        imagen_procesada = preparar_imagen(ruta)
        imagen_procesada = imagen_procesada.reshape(1, -1)  # Reshape para sklearn

        # Cargar modelo y predecir
        modelo = joblib.load('modelo_entrenado.pkl')
        prediccion = modelo.predict(imagen_procesada)

        etiqueta_resultado.config(text=f"Predicción: {prediccion[0]}")

# 🎨 Crear ventana
ventana = tk.Tk()
ventana.title("Clasificador de Imágenes")
ventana.geometry("300x400")

btn_cargar = tk.Button(ventana, text="Cargar Imagen", command=seleccionar_imagen)
btn_cargar.pack(pady=10)

panel = tk.Label(ventana)
panel.pack()

etiqueta_resultado = tk.Label(ventana, text="Predicción: ", font=("Arial", 16))
etiqueta_resultado.pack(pady=10)

# ⚙️ Entrenar modelo si no existe
entrenar_modelo()

ventana.mainloop()
