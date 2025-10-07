import cv2
import numpy as np
import streamlit as st

def run():
    st.header("🔷 Capítulo 8: Detector de Color (Azul)")
    
    scaling_factor = 0.5

    lower = np.array([60, 100, 100])
    upper = np.array([180, 255, 255])

    frame = cv2.imread('data/myphoto2.png')
    if frame is None:
        st.error("❌ No se pudo cargar la imagen. Verifica que exista 'data/myphoto2.png'")
        return
    
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,
                      interpolation=cv2.INTER_AREA)
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv_frame, lower, upper)
    
    res = cv2.bitwise_and(frame, frame, mask=mask)
    res = cv2.medianBlur(res, ksize=5)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(frame_rgb, caption='Original image', use_container_width=True)
    
    with col2:
        st.image(res_rgb, caption='Color Detector (Azul)', use_container_width=True)