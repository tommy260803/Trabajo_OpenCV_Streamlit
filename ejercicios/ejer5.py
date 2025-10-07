import cv2
import numpy as np
import streamlit as st

def run():
    st.header("🌫️ Capítulo 5: Detección de Características SIFT")
    
    input_image = cv2.imread('data/mar.png')
    
    if input_image is None:
        st.error("❌ No se pudo cargar la imagen. Verifica que exista 'data\mar.png'")
        return
    
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(gray_image, None)
    
    cv2.drawKeypoints(input_image, keypoints, input_image,
                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    st.image(input_image_rgb, caption='SIFT features', use_container_width=True)
    
    st.info(f"✅ Se detectaron {len(keypoints)} puntos característicos (keypoints)")