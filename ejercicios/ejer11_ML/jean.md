1)Objetivo: Clasificar imágenes como "gato" o "no gato"
2)Herramientas: Python, scikit-learn (para lo simple), o TensorFlow/Keras si quieres ir a algo más realista.
3)Dataset: Un conjunto pequeño de imágenes etiquetadas (gato/no gato). Puedes usar un dataset público o crear uno propio.
4)Pasos:
a) Preprocesamiento: Redimensionar imágenes, convertir a escala de grises, normalizar.
b) División del dataset: Entrenamiento y prueba.
c) Modelo: Usar un clasificador simple (como KNN o SVM) o una red neuronal básica.
d) Entrenamiento: Ajustar el modelo con los datos de entrenamiento.
e) Evaluación: Probar el modelo con los datos de prueba y medir precisión.
f) Predicción: Usar el modelo para clasificar nuevas imágenes.
5)Prueba: está en el script final de index.py
6)Donde se entrena el código:
mlp.fit(X_train, y_train)
