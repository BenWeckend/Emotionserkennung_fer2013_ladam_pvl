import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift

data = pd.read_csv('fer2013.csv')

pixels = data['pixels'].tolist()
labels = data['emotion'].tolist()

images = np.array([np.fromstring(pixel, dtype='int', sep=' ') for pixel in pixels])
images = images.reshape(-1, 48, 48, 1)  

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

def shift_images(images, labels, shift_pixels=1):
    shifted_images = []
    shifted_labels = []

    for image, label in zip(images, labels):
        shifted_left = shift(image[:, :, 0], [-shift_pixels, 0], cval=0)
        shifted_right = shift(image[:, :, 0], [shift_pixels, 0], cval=0)

        shifted_left = shifted_left.reshape(48, 48, 1)
        shifted_right = shifted_right.reshape(48, 48, 1)

        shifted_images.extend([shifted_left, shifted_right])
        shifted_labels.extend([label, label])

    shifted_images = np.array(shifted_images)
    shifted_labels = np.array(shifted_labels)

    return shifted_images, shifted_labels

X_train_shifted, y_train_shifted = shift_images(X_train, y_train)
X_test_shifted, y_test_shifted = shift_images(X_test, y_test)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

with tf.device('/GPU:0'):

    history = model.fit(X_train_shifted, y_train_shifted, epochs=10, validation_data=(X_test_shifted, y_test_shifted))

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss - Training vs. Validation')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')
plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy - Training vs. Validation')
plt.legend()

plt.tight_layout()
plt.show()

print("Anzahl der Parameter im Modell:")
print(model.count_params())
