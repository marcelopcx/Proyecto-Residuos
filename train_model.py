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

# ------------------------------------------------------------------
# PREPARACIÓN DE LOS DATOS CON AUMENTO Y PREPROCESAMIENTO
# ------------------------------------------------------------------

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Preprocesa las imágenes según MobileNetV2
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ------------------------------------------------------------------
# CONSTRUCCIÓN DEL MODELO USANDO TRANSFER LEARNING (MobileNetV2)
# ------------------------------------------------------------------

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
num_classes = train_generator.num_classes
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ------------------------------------------------------------------
# COMPILACIÓN DEL MODELO
# ------------------------------------------------------------------

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

steps_per_epoch = math.ceil(train_generator.samples / BATCH_SIZE)
validation_steps = math.ceil(validation_generator.samples / BATCH_SIZE)

# ------------------------------------------------------------------
# ENTRENAMIENTO DEL MODELO
# ------------------------------------------------------------------

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

# ------------------------------------------------------------------
# GUARDADO DEL MODELO ENTRENADO
# ------------------------------------------------------------------

model.save("waste_classifier_model.h5")
print("Modelo guardado en 'waste_classifier_model.h5'.")
