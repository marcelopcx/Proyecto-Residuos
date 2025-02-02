##############################################
#           Aplicación en Streamlit
##############################################

# Importar las librerías necesarias para la interfaz web y procesamiento de imágenes
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2

# ------------------------------------------------------------------
# FUNCIÓN PARA CARGAR EL MODELO ENTRENADO
# ------------------------------------------------------------------

@st.cache_resource
def load_model():
    # Cargar el modelo guardado durante el entrenamiento
    model = tf.keras.models.load_model("waste_classifier_model.h5")
    return model

# Se carga el modelo usando la función definida
model = load_model()

# ------------------------------------------------------------------
# DEFINICIÓN DE LAS ETIQUETAS DE CLASE
# ------------------------------------------------------------------

# Asegurarse de que el orden coincida con el utilizado durante el entrenamiento
class_labels = ['metalico', 'papel', 'plástico', 'vidrio']

# ------------------------------------------------------------------
# INTERFAZ DE USUARIO CON STREAMLIT
# ------------------------------------------------------------------

st.title("Clasificador de Residuos")
st.write("Suba una imagen y el sistema la clasificará.")

uploaded_file = st.file_uploader("Elija una imagen...", type=["jpg", "jpeg", "png"])
