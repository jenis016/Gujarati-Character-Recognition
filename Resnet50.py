from google.colab import drive
drive.mount('/content/drive')

import os
import tensorflow as tf
from tensorflow.keras import layers, models

import os
import cv2
import numpy as np

def decode_gujarati(string):
    return string.encode('utf-8').decode('utf-8')

dataset_dir = r'/content/drive/MyDrive/Dataset/Train'

class_directories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

images = []
labels = []

for class_dir in class_directories:
    class_label = decode_gujarati(class_dir)
    image_files = os.listdir(os.path.join(dataset_dir, class_dir))
    for image_file in image_files:
        image_path = os.path.join(dataset_dir, class_dir, image_file)
        images.append(image_path)
        labels.append(class_label)

images = np.array(images)
labels = np.array(labels)

print(f"Total number of images: {len(images)}")
print(f"Total number of labels: {len(labels)}")

print(labels)

import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


classes_dir = r'/content/drive/MyDrive/Dataset/Train'

class_dirs = os.listdir(classes_dir)

num_classes_to_display = min(7, len(class_dirs))

num_rows = (num_classes_to_display + 6) // 5 # Ensure at least 5 images per row

fig, axes = plt.subplots(num_rows, 7, figsize=(7, num_rows ))

for i, class_dir in enumerate(class_dirs):
    row = i // 7
    col = i % 7
    ax = axes[i] if num_rows == 1 else axes[row, col]
    image_files = [f for f in os.listdir(os.path.join(classes_dir, class_dir)) if f.endswith('.jpg') or f.endswith('.png')]
    if image_files:
        random_image = random.choice(image_files)
        img_path = os.path.join(classes_dir, class_dir, random_image)
        img = mpimg.imread(img_path)  # Read the image without any processing
        ax.imshow(img)
        ax.set_title(class_dir)
    else:
        ax.set_title(class_dir + " (No Images)")  # Indicate that there are no images in this directory
    ax.axis('off')

# Hide empty subplots
for i in range(num_classes_to_display, num_rows * 7):
    axes.flatten()[i].axis('off')

plt.tight_layout()
plt.show()

import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_image(image_path, output_dir):
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load image at", image_path)
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        target_height = 250
        target_width = int(aspect_ratio * target_height)
        resized_image = cv2.resize(image, (target_width, target_height))
        resized_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        _, binary_resized = cv2.threshold(resized_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, binary_resized)
    else:
        print("Error: No contours found in image at", image_path)


input_base_dir = r'/content/drive/MyDrive/Dataset/Train'
output_base_dir = r'/content/drive/MyDrive/Preprocessed Dataset'
label_encoder = LabelEncoder()
class_labels = []

for idx, class_dir in enumerate(sorted(os.listdir(input_base_dir))):
    class_labels.append(class_dir)
    label_encoder.fit_transform([class_dir])

for class_dir in os.listdir(input_base_dir):
    class_input_dir = os.path.join(input_base_dir, class_dir)
    class_output_dir = os.path.join(output_base_dir, class_dir)
    os.makedirs(class_output_dir, exist_ok=True)

    for filename in os.listdir(class_input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(class_input_dir, filename)
            preprocess_image(image_path, class_output_dir)

np.save('label_encoder.npy', label_encoder)

image_size = (112, 112)

batch_size = 32

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    output_base_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical', # Assuming you want one-hot encoded labels
    validation_split=0.2,
    subset="training",
    seed =42
)

data_augmentation_layers = [
    layers.RandomRotation(0.2, fill_mode='nearest'),
    layers.RandomZoom(0.3),
]

def data_augmentation(images, labels):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images, labels

augmented_dataset = dataset.map(data_augmentation)

pretrained_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=image_size + (3,),
    weights='imagenet'
)

# Freeze the layers of the pretrained model
for layer in pretrained_model.layers:
    layer.trainable = False

num_classes = 7

model = models.Sequential([
    pretrained_model,
    layers.Flatten(),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 100
history = model.fit(
    augmented_dataset,
    validation_data=augmented_dataset,
    epochs=epochs
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

true_labels = []
predicted_labels = []

for images, labels in dataset:
    true_labels.extend(tf.argmax(labels, axis=1))
    predictions = model.predict(images)
    predicted_labels.extend(tf.argmax(predictions, axis=1))

true_labels = tf.stack(true_labels, axis=0)
predicted_labels = tf.stack(predicted_labels, axis=0)

conf_matrix = confusion_matrix(true_labels, predicted_labels)

print("Confusion Matrix:")
print(conf_matrix)

class_names = dataset.class_names
report = classification_report(true_labels, predicted_labels,target_names=class_names)
print("Classification Report:")
print(report)

model.save('ResNet.h5')

