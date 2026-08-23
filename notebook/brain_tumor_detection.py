# # Brain Tumor Detection from MRI Images | A simple CNN model to detect brain tumors from MRI images.

# ## 1. Imports

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense


# ## 2. Load and Preprocess Data | Images are read, converted to RGB, and resized to 128x128. | Label 1 = tumor, Label 0 = no tumor.

IMG_SIZE = 128

def load_data(path_yes, path_no):
    data, labels = [], []
    for file in glob.glob(path_yes):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(1)
    for file in glob.glob(path_no):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(0)
    return np.array(data), np.array(labels)

data, labels = load_data(
    '../dataset/brain_tumor_dataset/yes/*',
    '../dataset/brain_tumor_dataset/no/*'
)
print('Total images:', len(data))
print('Tumor:', int(labels.sum()), '| No tumor:', int((labels == 0).sum()))


# ## 3. Normalize and Split Data | Pixel values are scaled to [0, 1] and the data is split into train (80%) and test (20%).

data = data / 255.0
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)
print('Train:', x_train.shape, 'Test:', x_test.shape)


# ## 4. Build the CNN Model

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# ## 5. Train the Model

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)


# ## 6. Plot Training History

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()


# ## 7. Evaluate the Model

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Accuracy:', round(test_acc * 100, 2), '%')


# ## 8. Save the Model

model.save('../model/brain_tumor_detection_model.h5')

