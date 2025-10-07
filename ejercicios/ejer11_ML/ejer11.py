import os
import numpy as np
from PIL import Image
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

TAMANO = (64, 64)  

def cargar_imagenes(carpeta, etiqueta):
    imagenes = []
    etiquetas = []
    if not os.path.isdir(carpeta):
        return imagenes, etiquetas
    for archivo in os.listdir(carpeta):
        ruta = os.path.join(carpeta, archivo)
        try:
            img = Image.open(ruta).convert('L').resize(TAMANO)  
            img_array = np.array(img).flatten() / 255.0
            imagenes.append(img_array)
            etiquetas.append(etiqueta)
        except Exception:
            print(f"Error al procesar: {ruta}")
    return imagenes, etiquetas


def load_dataset(base_path='data'):
    X_gato, y_gato = cargar_imagenes(os.path.join(base_path, 'gatos'), 1)
    X_no_gato, y_no_gato = cargar_imagenes(os.path.join(base_path, 'no_gatos'), 0)

    X = np.array(X_gato + X_no_gato)
    y = np.array(y_gato + y_no_gato)

    return X, y


def train_model(X, y, test_size=0.3, random_state=42):
    if len(X) == 0:
        raise ValueError('No hay imágenes en el dataset (carpetas vacías o rutas incorrectas).')

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=random_state)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['No Gato', 'Gato'])

    return mlp, acc, report


def predict_image(model, pil_image):
    img = pil_image.convert('L').resize(TAMANO)
    img_array = np.array(img).flatten() / 255.0
    pred = model.predict([img_array])[0]
    return int(pred)


def run():
    st.header('📚 Clasificador simple: gato / no gato')

    st.markdown('''
    Este ejercicio entrena un clasificador MLP sobre imágenes redimensionadas a 64x64.
    - Asegúrate de tener las carpetas `data/gatos` y `data/no_gatos` con imágenes.
    - Puedes entrenar el modelo y luego subir una imagen para predecir.
    - **Nota:** Subir la foto de un tiburón (pez)  para que no cometa un posible error de clasificación.
                esto porque no esta entrenado con miles de img.
    ''')

    X, y = load_dataset('data')
    n_samples = len(X)
    n_gatos = int(np.sum(y == 1)) if n_samples > 0 else 0
    n_no_gatos = int(np.sum(y == 0)) if n_samples > 0 else 0

    st.write(f'Dataset cargado: {n_samples} imágenes — Gatos: {n_gatos}, No gatos: {n_no_gatos}')

    model = None
    acc = None
    report = None

    model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')

    if 'mlp_model' not in st.session_state:
        if os.path.exists(model_path):
            try:
                st.session_state['mlp_model'] = joblib.load(model_path)
                st.success('Modelo cargado desde disco (model.joblib)')
            except Exception:
                pass

    if st.button('Entrenar modelo'):
        try:
            with st.spinner('Entrenando...'):
                model, acc, report = train_model(X, y)
            st.success(f'Modelo entrenado — Accuracy en test: {acc:.3f}')
            st.text(report)

            st.session_state['mlp_model'] = model
            try:
                joblib.dump(model, model_path)
                st.info(f'Modelo guardado en: {model_path}')
            except Exception as e:
                st.warning(f'No se pudo guardar el modelo en disco: {e}')
        except Exception as e:
            st.error(f'Error al entrenar: {e}')

    st.markdown('---')
    st.subheader('Probar una imagen')
    uploaded = st.file_uploader('Sube una imagen para clasificar (opcional)', type=['png', 'jpg', 'jpeg'])

    example_path = os.path.join('data', 'leopardo.jpg')

    if uploaded is not None:
        try:
            img = Image.open(uploaded)
            st.image(img, caption='Imagen subida', use_container_width=True)
        except Exception as e:
            st.error(f'No se pudo abrir la imagen subida: {e}')
            img = None
    else:
        if os.path.exists(example_path):
            img = Image.open(example_path)
            st.image(img, caption='Imagen de ejemplo (leopardo.jpg)', use_container_width=True)
        else:
            img = None

    if st.button('Eliminar modelo guardado'):
        if 'mlp_model' in st.session_state:
            del st.session_state['mlp_model']
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                st.success('Modelo guardado eliminado')
            except Exception as e:
                st.error(f'No se pudo eliminar el archivo de modelo: {e}')

    if st.button('Predecir imagen'):
        if img is None:
            st.error('No hay imagen disponible para predecir. Sube una o añade data/leopardo.jpg')
        else:
            model_to_use = st.session_state.get('mlp_model', None)
            if model_to_use is None and os.path.exists(model_path):
                try:
                    model_to_use = joblib.load(model_path)
                    st.session_state['mlp_model'] = model_to_use
                    st.info('Modelo cargado desde disco para la predicción')
                except Exception as e:
                    st.error(f'No se pudo cargar el modelo desde disco: {e}')

            if model_to_use is None:
                st.info('Entrena el modelo primero (presiona "Entrenar modelo").')
            else:
                try:
                    pred = predict_image(model_to_use, img)
                    st.success('SÍ' if pred == 1 else 'NO')
                    st.write('Predicción:', 'Gato' if pred == 1 else 'No Gato')
                except Exception as e:
                    st.error(f'Error en la predicción: {e}')

