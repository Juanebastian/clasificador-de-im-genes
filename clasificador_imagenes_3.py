import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# 🛠️ Entrenar el modelo si no existe
def entrenar_modelo():
    if not os.path.exists('modelo_entrenado.pkl'):
        print("Entrenando modelo...")
        digits = datasets.load_digits()
        X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
        modelo = LogisticRegression(max_iter=10000)
        modelo.fit(X_train, y_train)
        joblib.dump(modelo, 'modelo_entrenado.pkl')
        print("✅ Modelo guardado.")
    else:
        print("Modelo ya disponible.")

# 🎨 Funciones para dibujar
def dibujar(event):
    x, y = event.x, event.y
    r = 8  # Radio del punto
    lienzo.create_oval(x - r, y - r, x + r, y + r, fill='white', outline='white')
    draw.ellipse([x - r, y - r, x + r, y + r], fill='white')

def limpiar():
    lienzo.delete("all")
    draw.rectangle([0, 0, 200, 200], fill='black')
    etiqueta_resultado.config(text="Predicción: ")

def predecir():
    imagen_redimensionada = imagen.resize((8, 8))
    imagen_invertida = ImageOps.invert(imagen_redimensionada)  # Invertir para fondo negro
    imagen_array = np.array(imagen_invertida)
    imagen_array = (16 - (imagen_array / 16)).flatten()  # Escala similar al dataset

    modelo = joblib.load('modelo_entrenado.pkl')
    prediccion = modelo.predict([imagen_array])

    etiqueta_resultado.config(text=f"Predicción: {prediccion[0]}")

# 🎨 Crear ventana
ventana = tk.Tk()
ventana.title("Dibuja un Número")
ventana.geometry("300x400")
ventana.configure(bg='black')

# Lienzo para dibujar
lienzo = tk.Canvas(ventana, width=200, height=200, bg='black')
lienzo.pack(pady=10)
lienzo.bind("<B1-Motion>", dibujar)

# Imagen donde se guarda el dibujo
imagen = Image.new("L", (200, 200), color='black')
draw = ImageDraw.Draw(imagen)

# Botones
btn_predecir = tk.Button(ventana, text="Predecir", command=predecir)
btn_predecir.pack(pady=5)

btn_limpiar = tk.Button(ventana, text="Limpiar", command=limpiar)
btn_limpiar.pack(pady=5)

etiqueta_resultado = tk.Label(ventana, text="Predicción: ", font=("Arial", 16), fg='white', bg='black')
etiqueta_resultado.pack(pady=10)

# Entrenar modelo si es necesario
entrenar_modelo()

ventana.mainloop()
