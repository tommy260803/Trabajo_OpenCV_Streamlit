import streamlit as st
import cv2

def run():
    st.header("📖 Capítulo 1: Operaciones Básicas con Imágenes")
    
    img = cv2.imread('data/paisaje.png')
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    
    u_color = cv2.applyColorMap(v, cv2.COLORMAP_JET)

    
    v_color = cv2.applyColorMap(u, cv2.COLORMAP_JET)
    
    nuevo_tamaño = (500, 280)
    y_redim = cv2.resize(y, nuevo_tamaño)
    u_color_redim = cv2.resize(u_color, nuevo_tamaño)
    v_color_redim = cv2.resize(v_color, nuevo_tamaño)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(y_redim, caption='Canal Y (BRILLO REAL)', channels='GRAY')
    with col2:
        st.image(u_color_redim, caption='Canal U (Componente Azul-Amarillo COLOR)', channels='BGR')
    with col3:
        st.image(v_color_redim, caption='Canal V (Componente Rojo-Verde COLOR)', channels='BGR')