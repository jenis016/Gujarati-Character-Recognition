from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to preprocess images
def preprocess_image(image_path, target_size=(28, 28)):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize image
    image = cv2.resize(image, target_size)
    # Normalize pixel values to range [0, 1]
    image = image.astype('float32') / 255.0
    # Expand dimensions to match expected input shape for neural network
    image = np.expand_dims(image, axis=-1)
    return image

# Function to load and preprocess images from a directory
def load_images_from_directory(directory, target_size=(28, 28)):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                image = preprocess_image(image_path, target_size)
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

# Directory containing handwritten character images
data_dir = '/content/drive/MyDrive/Test_Dataset'

# Load and preprocess images
images, labels = load_images_from_directory(data_dir)

# Convert labels to categorical (one-hot encoding)
from tensorflow.keras.utils import to_categorical
label_to_index = {label: idx for idx, label in enumerate(sorted(set(labels)))}
labels = np.array([label_to_index[label] for label in labels])
labels = to_categorical(labels, num_classes=len(label_to_index))

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_to_index), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=2)
print(f"Validation accuracy: {val_acc}")

# Save the model
model.save('handwritten_character_recognition_model.h5')

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Load the saved model
model = tf.keras.models.load_model('handwritten_character_recognition_model.h5')

def predict_image(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)
    # Add a batch dimension
    image = np.expand_dims(image, axis=0)
    # Make a prediction
    prediction = model.predict(image)
    # Get the predicted label
    predicted_label = np.argmax(prediction, axis=1)
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    return index_to_label[predicted_label[0]]

# Test the model with a new image
new_image_path = '/content/છ (20).png'
predicted_label = predict_image(new_image_path)
print(f"The predicted label for the new image is: {predicted_label}")