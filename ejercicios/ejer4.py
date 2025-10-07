import cv2
import numpy as np
import streamlit as st

def run():
    st.header("🎨 Capítulo 4: Detector de Ojos")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    if face_cascade.empty():
        st.error('❌ No se pudo cargar el clasificador de rostros')
        return
    if eye_cascade.empty():
        st.error('❌ No se pudo cargar el clasificador de ojos')
        return
    
    frame = cv2.imread('data/MyPhoto.png')
    if frame is None:
        st.error("❌ No se pudo cargar la imagen. Verifica que exista 'data/MyPhoto.png'")
        return
    
    ds_factor = 0.5
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,
                      interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            eye_center_y = y_eye + h_eye // 2
            if eye_center_y > int(0.6 * roi_gray.shape[0]):
                continue
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            color = (0, 255, 0)
            thickness = 3
            cv2.circle(roi_color, center, radius, color, thickness)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption='Eye Detector', use_container_width=True)
    
    st.info(f"✅ Se detectaron {len(faces)} rostro(s)")