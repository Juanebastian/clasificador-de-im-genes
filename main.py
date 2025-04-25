from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import numpy as np

# Cargar el dataset de dígitos
digits = datasets.load_digits()

# Guardar imagen de ejemplo
plt.gray()
plt.matshow(digits.images[0])
plt.title(f"Etiqueta: {digits.target[0]}")
plt.savefig('ejemplo.png')
plt.close()

# Preparación de datos
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Resultados
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

# Guardar predicciones
for i in range(5):
    image = np.reshape(X_test[i], (8, 8))
    plt.matshow(image, cmap='gray')
    plt.title(f"Etiqueta real: {y_test[i]} - Predicción: {y_pred[i]}")
    plt.savefig(f'prediccion_{i}.png')
    plt.close()