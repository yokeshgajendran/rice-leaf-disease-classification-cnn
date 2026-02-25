# ==========================================================
# RICE LEAF DISEASE CLASSIFICATION USING CNN
# ==========================================================

# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ==========================================================
# 2. DATA PREPARATION
# ==========================================================
# Dataset folder structure:
# dataset/
#    ├── Bacterialblight
#    ├── Blast
#    ├── Brownspot
#    └── Tungro

train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

img_width, img_height = 150, 150
batch_size = 32

# ==========================================================
# 3. DATA AUGMENTATION
# ==========================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical')

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical')

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical')

# ==========================================================
# 4. BUILD CNN MODEL
# ==========================================================

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==========================================================
# 5. MODEL TRAINING
# ==========================================================

Model_CNN = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size)

model.save('Custom_CNN_Model_1.h5')

# ==========================================================
# 6. MODEL EVALUATION
# ==========================================================

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)

# ==========================================================
# 7. CLASSIFICATION REPORT
# ==========================================================

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('Custom_CNN_Model_1.h5')
test_images_dir = 'dataset/test'
test_image_files = []
for subdir, dirs, files in os.walk(test_images_dir):
  for file in files:
    test_image_files.append(os.path.join(subdir, file))

class_labels = list(train_generator.class_indices.keys())
label_to_index = {label: index for index, label in enumerate(class_labels)}

true_labels = []
predicted_labels = []

for img_path in test_image_files:
    if os.path.isfile(img_path):
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        true_label = os.path.basename(os.path.dirname(img_path))
        true_class_index = label_to_index.get(true_label, -1)

        if true_class_index != -1:
            true_labels.append(true_class_index)
            predicted_labels.append(predicted_class_index)
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_labels))

# ==========================================================
# 7. VISUAL COMPARISON (OPTIONAL)
# ==========================================================

test_images_dir = 'dataset/test'

class_labels = list(train_generator.class_indices.keys())
images_per_class = {}
for subdir, dirs, files in os.walk(test_images_dir):
    class_name = os.path.basename(subdir)
    if class_name in class_labels and class_name not in images_per_class:
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                images_per_class[class_name] = os.path.join(subdir, file)
                break

for class_name, img_path in images_per_class.items():
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f'Actual: {class_name}, Predicted: {predicted_class}')
    plt.axis('off')
    plt.show()