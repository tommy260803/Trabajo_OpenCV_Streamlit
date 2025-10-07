import cv2
import numpy as np
import streamlit as st

class DenseDetector():
    def __init__(self, step_size=20, feature_scale=20, img_bound=20):
        self.initXyStep = step_size
        self.initFeatureScale = feature_scale
        self.initImgBound = img_bound
    
    def detect(self, img):
        keypoints = []
        rows, cols = img.shape[:2]
        for x in range(self.initImgBound, rows, self.initFeatureScale):
            for y in range(self.initImgBound, cols, self.initFeatureScale):
                keypoints.append(cv2.KeyPoint(float(x), float(y), self.initXyStep))
        return keypoints

class SIFTDetector():
    def __init__(self):
        self.detector = cv2.xfeatures2d.SIFT_create()
    
    def detect(self, img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(gray_image, None)

def run():
    st.header("🎯 Capítulo 9: Comparación Dense vs SIFT Detector")
    
    input_image = cv2.imread('data/tren.png')
    
    if input_image is None:
        st.error("❌ No se pudo cargar la imagen. Verifica que exista 'data/tren.png'")
        return
    
    input_image_dense = np.copy(input_image)
    input_image_sift = np.copy(input_image)
    
    keypoints = DenseDetector(20, 20, 5).detect(input_image)
    input_image_dense = cv2.drawKeypoints(input_image_dense, keypoints, None,
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    keypoints_sift = SIFTDetector().detect(input_image)
    input_image_sift = cv2.drawKeypoints(input_image_sift, keypoints_sift, None,
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    input_image_dense_rgb = cv2.cvtColor(input_image_dense, cv2.COLOR_BGR2RGB)
    input_image_sift_rgb = cv2.cvtColor(input_image_sift, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(input_image_dense_rgb, caption='Dense feature detector', use_container_width=True)
        st.info(f"Keypoints detectados: {len(keypoints)}")
    
    with col2:
        st.image(input_image_sift_rgb, caption='SIFT detector', use_container_width=True)
        st.info(f"Keypoints detectados: {len(keypoints_sift)}")