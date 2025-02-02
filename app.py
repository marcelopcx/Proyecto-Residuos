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

if uploaded_file is not None:
    # Convertir el archivo cargado a un arreglo de bytes y decodificarlo a una imagen
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convertir de BGR a RGB para mostrar correctamente la imagen
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption='Imagen subida.', use_container_width=True)

    # ------------------------------------------------------------------
    # PREPROCESAMIENTO DE LA IMAGEN PARA LA PREDICCIÓN
    # ------------------------------------------------------------------

    image_resized = cv2.resize(image, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0)
    image_array = preprocess_input(image_array)

    # ------------------------------------------------------------------
    # REALIZAR LA PREDICCIÓN CON EL MODELO
    # ------------------------------------------------------------------

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    st.write(f"**Predicción:** {class_labels[predicted_class]}")
    st.write(f"**Confianza:** {confidence * 100:.2f}%")

# (Opcional) Ajuste final de la interfaz: agregar mensaje informativo si no se sube imagen
if uploaded_file is None:
    st.info("Por favor, suba una imagen para clasificarla.")
