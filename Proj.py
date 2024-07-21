import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation (optional) to improve generalization
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Prepare training and validation data generators
train_generator = train_datagen.flow_from_directory(
    r"C:\Users\yadav\Desktop\Priyanka\content\Dataset\Train",
    target_size=(150,150),  # Adjust for your model's input size
    batch_size=64,  # Experiment with batch size
    class_mode='binary'
)
validation_generator = validation_datagen.flow_from_directory(
    r"C:\Users\yadav\Desktop\Priyanka\content\Dataset\Valid",
    target_size=(150,150),
    batch_size=64,
    class_mode='binary'
)

from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Dropout
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras.applications import Xception

# Input Layer
input_layer = Input(shape=(150,150, 3))

# Inception Module
base_model = Xception(weights='imagenet', include_top=False, input_tensor=input_layer, classes=1)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.1)(x)
output_layer = Dense(1, activation='sigmoid')(x)

# Model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Freeze the layers of model

base_model.trainable = True
set_trainable = False
for layer in base_model.layers:
    if layer.name == 'block14_sepconv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

for layer in model.layers:
    print(layer.name, layer.trainable)

model.summary()

from tensorflow.keras.callbacks import EarlyStopping

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights=True)

# Train the model with early stopping
model.fit(
    train_generator,
    epochs=10,  # Adjust based on dataset size and validation performance
    validation_data=validation_generator,
    callbacks=[early_stopping]
)
model.save("Project.h5")
