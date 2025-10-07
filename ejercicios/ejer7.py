import cv2
import numpy as np
import streamlit as st

def get_all_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def run():
    st.header("🧩 Capítulo 7: Detección y Suavizado de Contornos")
    
    img1 = cv2.imread('data/image.png')
    
    if img1 is None:
        st.error("❌ No se pudo cargar la imagen. Verifica que exista 'data/image.png'")
        return
    
    factor = st.slider("Factor de suavizado:", min_value=0.001, max_value=0.5, value=0.05, step=0.001, format="%.3f")
    
    st.info(f"**Factor actual:** {factor} - Menor valor = más detalle, Mayor valor = más simplificado")
    
    input_contours = get_all_contours(img1)
    contour_img = img1.copy()
    smoothen_contours = []
    
    for contour in input_contours:
        epsilon = factor * cv2.arcLength(contour, True)
        smoothen_contours.append(cv2.approxPolyDP(contour, epsilon, True))
    
    cv2.drawContours(contour_img, smoothen_contours, -1, color=(0, 0, 0), thickness=3)
    
    contour_img_rgb = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
    
    st.image(contour_img_rgb, caption='Contours', use_container_width=True)