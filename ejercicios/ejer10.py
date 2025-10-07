import cv2
import numpy as np
import streamlit as st
from PIL import Image

class ObjectTrackerCSRT:
    def __init__(self):
        self.tracker = None
        self.initialized = False

    def init_tracker(self, frame, rect):
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, rect)
        self.initialized = True

    def update(self, frame):
        if not self.initialized:
            return None
        success, box = self.tracker.update(frame)
        return box if success else None

def run():
    st.header("📘 Seguimiento de un libro en video (Tracker CSRT)")
    
    st.markdown("""
    **Instrucciones:**
    1. Ingresamos las coordenadas del libro (x, y, ancho, alto) esta abajo del video.
    2. Seleccionar x=40, y =200, ancho y largo = 50.
    3. Presionar **Procesar Video** para ver cómo el sistema sigue el objeto.
    """)


    video_path = 'data/video.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ No se pudo cargar el video. Verifica que exista 'data/video.mp4'")
        return

    scaling_factor = 0.6

    ret, frame = cap.read()
    if not ret:
        st.error("❌ Error al leer el video")
        cap.release()
        return

    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption="Primer frame del video", use_container_width=True)

    st.markdown("### 📍 Define el área del libro a rastrear:")
    col1, col2, col3, col4 = st.columns(4)
    x = col1.number_input("X:", min_value=0, max_value=frame.shape[1], value=100)
    y = col2.number_input("Y:", min_value=0, max_value=frame.shape[0], value=100)
    w = col3.number_input("Ancho:", min_value=10, max_value=frame.shape[1], value=100)
    h = col4.number_input("Alto:", min_value=10, max_value=frame.shape[0], value=100)

    rect = (x, y, w, h)

    if st.button("Procesar Video"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)

        tracker = ObjectTrackerCSRT()
        tracker.init_tracker(frame, rect)

        output_frames = []
        frame_count = 0
        max_frames = 300
        progress = st.progress(0)

        while ret and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)
            box = tracker.update(frame)

            if box is not None:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "estoy aqui...", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Objeto perdido", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            output_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1
            progress.progress(frame_count / max_frames)

        cap.release()

        st.success(f"✅ Procesados {frame_count} frames")
        st.markdown("### Ejemplo de frames rastreados:")
        cols = st.columns(3)
        for i, idx in enumerate([0, len(output_frames)//2, len(output_frames)-1]):
            if idx < len(output_frames):
                with cols[i]:
                    st.image(output_frames[idx], caption=f"Frame {idx+1}", use_container_width=True)
