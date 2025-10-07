import cv2
import numpy as np
import streamlit as st

def cartoonize_image(img, ksize=5, sketch_mode=False):
    num_repetitions, sigma_color, sigma_space, ds_factor = 10, 5, 7, 4
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_gray = cv2.medianBlur(img_gray, 7)
    
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
   
    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
   
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor,
                          interpolation=cv2.INTER_AREA)
   
    for i in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_color,
                                        sigma_space)
    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor,
                           interpolation=cv2.INTER_LINEAR)
    dst = np.zeros(img_gray.shape)
    
    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    return dst

def run():
    st.header("🔄 Capítulo 3: Cartoonize Image")
    
    st.markdown("""
    **Modos de cartoonización:**
    - **Sketch Mode (Sin Color):** Solo contornos en blanco y negro
    - **Color Mode (Con Color):** Imagen con efecto cartoon a color
    """)
    
    # Selector de modo
    mode = st.radio("Selecciona el modo:", ["Original", "Cartoonize sin Color", "Cartoonize con Color"])
    
    # Cargar imagen de prueba
    img = cv2.imread('data/MyPhoto.png')
    
    if img is None:
        st.error("❌ No se pudo cargar la imagen. Verifica que exista en 'data/'")
        return
    
    # Redimensionar para procesamiento más rápido
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
    # Aplicar el efecto según el modo seleccionado
    if mode == "Cartoonize sin Color":
        result = cartoonize_image(img, ksize=5, sketch_mode=True)
    elif mode == "Cartoonize con Color":
        result = cartoonize_image(img, ksize=5, sketch_mode=False)
    else:
        result = img
    
    # Convertir BGR a RGB para mostrar
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # Mostrar resultado
    st.image(result_rgb, caption=mode, use_container_width=600)