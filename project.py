# Import the necessary libraries
import tensorflow as tf
import keras
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Data augmentation (optional) to improve generalization
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Prepare training and validation data generators
train_generator = train_datagen.flow_from_directory(
    r"C:\Users\yadav\Desktop\Priyanka\content\Dataset\Train",
    target_size=(224, 224),  # Adjust for your model's input size
    batch_size=64,  # Experiment with batch size
    class_mode='binary'
)
validation_generator = validation_datagen.flow_from_directory(
    r"C:\Users\yadav\Desktop\Priyanka\content\Dataset\Valid",
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary'
)

# Define a function to create a CNN model for deepfake detection using tensorflow.keras.Sequential
def create_model():
    # Create an empty sequential model
    model = Sequential()
    # Add a convolutional layer with 32 filters, 3 x 3 kernel size, ReLU activation and input shape of (224, 224, 3)
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)))
    # Add a max pooling layer with 2 x 2 pool size
    model.add(MaxPooling2D((2, 2)))
    # Add a convolutional layer with 64 filters, 3 x 3 kernel size and ReLU activation
    model.add(Conv2D(64, (3, 3), activation="relu"))
    # Add a max pooling layer with 2 x 2 pool size
    model.add(MaxPooling2D((2, 2)))
    # Add a convolutional layer with 128 filters, 3 x 3 kernel size and ReLU activation
    model.add(Conv2D(128, (3, 3), activation="relu"))
    # Add a max pooling layer with 2 x 2 pool size
    model.add(MaxPooling2D((2, 2)))
    # Add a flatten layer to convert the 3D feature maps to 1D feature vectors
    model.add(Flatten())
    # Add a dense layer with 256 units and ReLU activation
    model.add(Dense(256, activation="relu"))
    # Add a dropout layer with 0.5 dropout rate to prevent overfitting
    model.add(Dropout(0.5))
    # Add a dense layer with 1 unit and sigmoid activation for binary classification
    model.add(Dense(1, activation="sigmoid"))
    # Return the model
    return model

# Create the CNN model
model = create_model()
model.summary()

# Compile the model using Adam optimizer, binary crossentropy loss and accuracy metric
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

keras.saving.save_model(model, 'best_model.keras')

from tensorflow.keras.callbacks import EarlyStopping

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights=True)

# Train the model with early stopping
history = model.fit(
    train_generator,
    epochs=10,  # Adjust based on dataset size and validation performance
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(loc='lower right')
plt.title('Training History')
plt.show()

model.save("Minipre.keras")
model.save("Minipre.h5")
keras.saving.save_model(model, 'Minimodel.keras')

