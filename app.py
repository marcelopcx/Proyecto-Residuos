##############################################
#           Aplicación en Streamlit
##############################################

# Importar las librerías necesarias para la interfaz web y procesamiento de imágenes
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2