import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,  # pyright: ignore[reportUnusedImport, reportUnusedImport]
    Dropout
)

# Dataset path
DATASET_PATH = r"C:\Users\dhanu\OneDrive\Desktop\IIIT GRAND FINAL\Project\datasets\TESS Toronto emotional speech set data"

# Lists
X = []
y = []

# Load dataset
for emotion_folder in os.listdir(DATASET_PATH):

    emotion_path = os.path.join(DATASET_PATH, emotion_folder)

    if not os.path.isdir(emotion_path):
        continue

    emotion = emotion_folder.split("_")[-1].lower()

    

# Fix inconsistent labels
    if emotion == "surprised":
        emotion = "surprise"

    for file_name in os.listdir(emotion_path):

        if file_name.endswith(".wav"):

            file_path = os.path.join(emotion_path, file_name)

            try:
                # Load audio
                audio, sr = librosa.load(file_path, sr=16000)

                # Trim silence
                audio, _ = librosa.effects.trim(audio)

                # Extract MFCC
                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=sr,
                    n_mfcc=40
                )

                # Mean MFCC
                mfcc = mfcc.T

                X.append(mfcc)
                y.append(emotion)

            except Exception as e:
                print("Error:", file_path)
                print(e)

# Convert to numpy arrays
# Pad sequences
X = pad_sequences(
    X,
    padding='post',
    dtype='float32'
)

# Convert labels
y = np.array(y)

print("Padded X Shape:", X.shape)
print("y Shape:", y.shape)

print("Dataset Loaded!")
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# Encode labels
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)
print(encoder.classes_)


# Convert labels to categorical
y_categorical = to_categorical(y_encoded)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# Build model
model = Sequential()

# CNN Layer
model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        input_shape=(X.shape[1], X.shape[2])
    )
)

# Pooling
model.add(MaxPooling1D(pool_size=2))

# BiLSTM
model.add(
    Bidirectional(
        LSTM(64)
    )
)

# Dropout
model.add(Dropout(0.3))

# Dense Layer
model.add(Dense(64, activation='relu'))

# Output Layer
model.add(
    Dense(
        y_categorical.shape[1],
        activation='softmax'
    )
)

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

# Save Accuracy Plot
plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Speech Model Accuracy')
plt.savefig("../../Results/plots/speech_accuracy.png")
plt.close()

# Save Loss Plot
plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Speech Model Loss')
plt.savefig("../../Results/plots/speech_loss.png")
plt.close()


print("\nTest Accuracy:", accuracy)

# Save model
model.save("speech_emotion_model.h5")

print("\nModel Saved Successfully!")