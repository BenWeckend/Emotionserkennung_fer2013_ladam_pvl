# Ladam_pvl / Ben Weckend MatrN: 67551
Fertiges Programm: main6.py in Ordner: src1

- Test Loss: 0.7174373269081116
- Test Accuracy: 0.7609543800354004
- Anzahl der Parameter im Modell: 944487


Zunächst importiert man die erforderlichen Bibliotheken:
``` python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy.ndimage import shift
```

Klasse EmotionClassifier wird definiert, diese enthält die Funktionalitäten zur Emotionserkennung:
``` python
class EmotionClassifier:
    def __init__(self):
        self.model = self.build_model()
```
Datensatz auslesen mit pandas:
``` python
    @staticmethod
    def read_data(filename):
        data = pd.read_csv(filename)
        return data
```
Umwandeln der Daten in 2D pixel Werte:
``` python
    @staticmethod
    def preprocess_data(data):
        pixels = data['pixels'].tolist()
        labels = data['emotion'].tolist()
        images = np.array([np.fromstring(pixel, dtype='int', sep=' ') for pixel in pixels]).reshape(-1, 48, 48, 1)
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        return images, labels
```
Erweiterung des Datensatzes durch spiegeln und Verschiebung der Bilder um einen Pixel um jeweil rechts und links:
``` python
    @staticmethod
    def augment_data(images, labels):
        augmented_images = []
        augmented_labels = []

        for image, label in zip(images, labels):
            # Vertikal spiegeln
            flipped_image = np.flip(image, axis=0)
            augmented_images.extend([image, flipped_image])
            augmented_labels.extend([label, label])

            # Horizontal shift
            shifted_left = shift(image[:, :, 0], [0, -1], cval=0)
            shifted_right = shift(image[:, :, 0], [0, 1], cval=0)
            shifted_left = shifted_left.reshape(48, 48, 1)
            shifted_right = shifted_right.reshape(48, 48, 1)
            augmented_images.extend([shifted_left, shifted_right])
            augmented_labels.extend([label, label])

        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)

        return augmented_images, augmented_labels
```
Teilen des Datensatzes in Training- und Testdaten:
``` python
    @staticmethod
    def split_data(images, labels, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
```
Modell erstellen:
``` python
    def build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(7, activation='softmax')
        ])
        model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
```
Modell trainiert:
``` python
    def train_model(self, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        return history
```
Berechnen von loss und accuracy:
``` python

    def evaluate_model(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy
```
Daten plotten:
``` python
    @staticmethod
    def plot_loss_accuracy(history):
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
```
Funktion zum zählen der Parameter und speichern des Models:
``` python
    def count_parameters(self):
        return self.model.count_params()

    def save_model(self, filename):
        self.model.save(filename)

```
Aufrufen von Funktionen + Rest:
``` python

emotion_classifier = EmotionClassifier()
data = emotion_classifier.read_data('fer2013.csv')
images, labels = emotion_classifier.preprocess_data(data)

augmented_images, augmented_labels = emotion_classifier.augment_data(images, labels)

X_train, X_test, y_train, y_test = emotion_classifier.split_data(augmented_images, augmented_labels)

history = emotion_classifier.train_model(X_train, y_train, X_test, y_test, epochs=5)
loss, accuracy = emotion_classifier.evaluate_model(X_test, y_test)

print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
emotion_classifier.plot_loss_accuracy(history)
print("Anzahl der Parameter im Modell:")
print(emotion_classifier.count_parameters())


emotion_classifier.save_model("model.h5")
```
