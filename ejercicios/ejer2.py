import cv2
import streamlit as st
import numpy as np

def run():
    st.header("✏️ Capítulo 2: Detección de Bordes con Sobel")
    
    img = cv2.imread('data/tren.png', cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape
    
    
    sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

    sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    
   
    sobel_horizontal = np.uint8(np.absolute(sobel_horizontal))
    sobel_vertical = np.uint8(np.absolute(sobel_vertical))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img, caption='Original', channels='GRAY')
    with col2:
        st.image(sobel_horizontal, caption='Sobel horizontal', channels='GRAY')
    with col3:
        st.image(sobel_vertical, caption='Sobel vertical', channels='GRAY')