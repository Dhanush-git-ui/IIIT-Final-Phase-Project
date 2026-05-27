import os
import numpy as np
import librosa
import torch
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertModel

# =========================================================
# CONFIGURATION
# =========================================================
DATASET_PATH = r"C:\Users\dhanu\OneDrive\Desktop\IIIT GRAND FINAL\Project\datasets\TESS Toronto emotional speech set data"
SPEECH_MODEL_PATH = r"C:\Users\dhanu\OneDrive\Desktop\IIIT GRAND FINAL\Project\models\speech_pipeline\speech_emotion_model.h5"

# =========================================================
# LOAD SPEECH MODEL & BERT
# =========================================================
print("Loading Speech Model...")
speech_model = load_model(SPEECH_MODEL_PATH)
print("Speech Model Loaded!")

speech_feature_extractor = Model(
    inputs=speech_model.inputs,
    outputs=speech_model.layers[-2].output
)
print("Speech Feature Extractor Ready!")

print("Loading BERT...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
print("BERT Loaded!")

# =========================================================
# LOAD TESS DATASET & EXTRACT RAW FEATURES
# =========================================================
print("Loading TESS Dataset...")
X_speech_raw = []
texts = []
labels_raw = []

emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
emotion_to_idx = {emo: i for i, emo in enumerate(emotions)}

# Iterate through dataset folders
for emotion_folder in os.listdir(DATASET_PATH):
    emotion_path = os.path.join(DATASET_PATH, emotion_folder)
    if not os.path.isdir(emotion_path):
        continue
    
    emotion = emotion_folder.split("_")[-1].lower()
    if emotion == "surprised":
        emotion = "surprise"
        
    if emotion not in emotion_to_idx:
        continue
        
    print(f"Processing folder: {emotion_folder} (mapped to '{emotion}')")
    
    for file_name in os.listdir(emotion_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(emotion_path, file_name)
            try:
                # Load audio
                audio, sr = librosa.load(file_path, sr=16000)
                
                # Trim silence
                audio, _ = librosa.effects.trim(audio)
                
                # Extract MFCC
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                mfcc = mfcc.T
                
                # Extract word from filename
                parts = file_name.split("_")
                word = parts[1] if len(parts) >= 2 else "word"
                
                # Construct sentence spoken in the audio
                phrase = f"Say the word {word}."
                
                X_speech_raw.append(mfcc)
                texts.append(phrase)
                labels_raw.append(emotion_to_idx[emotion])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

print(f"Loaded {len(X_speech_raw)} samples successfully!")

# =========================================================
# SPEECH FEATURE EXTRACTION (EMBEDDINGS)
# =========================================================
print("Padding speech sequences...")
# Pad speech sequences to length 94 to match the expected shape (None, 94, 40)
X_speech_padded = pad_sequences(
    X_speech_raw,
    maxlen=94,
    padding='post',
    dtype='float32'
)

print("Extracting Speech Embeddings...")
speech_embeddings = speech_feature_extractor.predict(X_speech_padded, batch_size=64)
print("Speech Embeddings Extracted Shape:", speech_embeddings.shape)

# =========================================================
# TEXT FEATURE EXTRACTION (BERT EMBEDDINGS)
# =========================================================
print("Extracting Text Embeddings via BERT...")
text_embeddings_list = []
batch_size = 64

# Put BERT model in evaluation mode
bert_model.eval()

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i : i + batch_size]
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.pooler_output.numpy()
    text_embeddings_list.append(embeddings)

text_embeddings = np.concatenate(text_embeddings_list, axis=0)
print("Text Embeddings Extracted Shape:", text_embeddings.shape)

# =========================================================
# CONCATENATE FEATURES & ALIGN LABELS
# =========================================================
fusion_features = np.concatenate([speech_embeddings, text_embeddings], axis=1)
print("Fusion Feature Shape:", fusion_features.shape)

labels = to_categorical(np.array(labels_raw), num_classes=7)
print("Labels Shape:", labels.shape)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    fusion_features,
    labels,
    test_size=0.2,
    random_state=42
)
print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# =========================================================
# BUILD FUSION MODEL
# =========================================================
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(fusion_features.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nFusion Model Final Test Accuracy: {accuracy:.4f}")

# Save Accuracy and Loss Plots
import matplotlib.pyplot as plt

os.makedirs("../../Results/plots", exist_ok=True)

# Save Accuracy Plot
plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Fusion Model Accuracy')
plt.savefig("../../Results/plots/fusion_accuracy.png")
plt.close()

# Save Loss Plot
plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Fusion Model Loss')
plt.savefig("../../Results/plots/fusion_loss.png")
plt.close()

# Save the trained model
model.save("fusion_model.keras")
print("\nFusion Model Saved Successfully!")