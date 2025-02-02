##############################################
#        Entrenamiento y Evaluación del Modelo
##############################################

# Importar librerías necesarias
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------------
# CONFIGURACIÓN Y PARÁMETROS DEL ENTRENAMIENTO
# ------------------------------------------------------------------

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

TRAIN_DIR = "data/train"
VALIDATION_DIR = "data/validation"