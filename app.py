import streamlit as st

from ejercicios import ejer1, ejer2, ejer3, ejer4, ejer5, ejer6, ejer7, ejer8, ejer9, ejer10
from ejercicios.ejer11_ML import ejer11 as ejer11

st.set_page_config(
    page_title="Trabajo OpenCV - Streamlit",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

col1, col2 = st.columns([3, 1])
with col1:
    st.title("📷 Trabajo de OpenCV")
    st.markdown("### Ejercicios del libro: *OpenCV 3.x with Python By Example*")
    st.markdown("---")
with col2:
    st.image("https://opencv.org/wp-content/uploads/2020/07/OpenCV_logo_no_text_.png", width=200)

st.subheader("Este trabajo fue realizado por el estudiante: ")

st.markdown(
    "<span style='font-size: 42px; font-family:Geneva; font-weight: bold;'>"
    "<span style='color: red;'>Anthony</span> "
    "<span style='color: green;'>Garcia</span> "
    "<span style='color: blue;'>Ventura</span>"
    "</span>",
    unsafe_allow_html=True
)

st.sidebar.title("🗂️ Menú de Ejercicios")
st.sidebar.markdown("Selecciona un capítulo para visualizar:")

capitulos = {
    "🏠 Inicio": None,
    "📖 Cap 1: Operaciones Básicas con Imágenes": ejer1,
    "✏️ Cap 2: Detección de Bordes con Sobel": ejer2,
    "🔄 Cap 3: Cartoonize Image": ejer3,
    "🎨 Cap 4: Detector de Ojos": ejer4,
    "🌫️ Cap 5: Detección de Características SIFT": ejer5,
    "🔍 Cap 6: Detección de Bordes": ejer6,
    "🧩 Cap 7: Detección y Suavizado de Contornos": ejer7,
    "🔷 Cap 8: Detector de Color (Azul)": ejer8,
    "🎯 Cap 9: Comparación Dense vs SIFT Detector": ejer9,
    "⭐ Cap 10:  Seguimiento de un libro en video (Tracker CSRT)": ejer10,
    "🚀 Cap 11: Entrenamiento IA || ML": ejer11
}

opcion = st.sidebar.selectbox(
    "Elige un capítulo:",
    list(capitulos.keys())
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Anthony Garcia** 
                
**Curso:** Sistemas Inteligentes  
**Profesor:** Ing. Torres Villanueva Marcelino
**Libro:** OpenCV 3.x with Python By Example  
**Autores:** Gabriel Garrido & Prateek Joshi
""")

# Contenido principal
if opcion == "🏠 Inicio":
    st.markdown("""
    ## Bienvenido ING
    
    Este proyecto contiene **11 ejercicios** basados en el libro **OpenCV 3.x with Python By Example**.
    
    ### 📚 Contenido del trabajo:
    
    - **Capítulo 1:** Operaciones básicas con imágenes (cargar, mostrar, guardar)
    - **Capítulo 2:** Dibujo de figuras geométricas
    - **Capítulo 3:** Transformaciones geométricas (rotación, escala, traslación)
    - **Capítulo 4:** Manipulación de píxeles y canales de color
    - **Capítulo 5:** Aplicación de filtros y desenfoques
    - **Capítulo 6:** Detección de bordes (Canny, Sobel, etc.)
    - **Capítulo 7:** Segmentación de imágenes
    - **Capítulo 8:** Operaciones morfológicas (erosión, dilatación)
    - **Capítulo 9:** Detección de objetos
    - **Capítulo 10:** Detección de características y descriptores
    - **Capítulo 11:** Proyecto final integrador
    
    ### 🎯 Instrucciones:
    
    Utilizar el **menú lateral izquierdo** para navegar entre los diferentes capítulos.
    
    ---

    💡 **OJO** Algunos ejercicios tienen funcionalidades interactivas.
    """)
    

else:
    modulo = capitulos[opcion]
    if modulo:
        modulo.run()
    else:
        st.warning("Este capítulo aún no está implementado.")