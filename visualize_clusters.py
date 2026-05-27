import os
import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertModel

# Configure Paths
DATASET_PATH = r"C:\Users\dhanu\OneDrive\Desktop\IIIT GRAND FINAL\Project\datasets\TESS Toronto emotional speech set data"
SPEECH_MODEL_PATH = r"C:\Users\dhanu\OneDrive\Desktop\IIIT GRAND FINAL\Project\models\speech_pipeline\speech_emotion_model.h5"
PLOTS_DIR = r"C:\Users\dhanu\OneDrive\Desktop\IIIT GRAND FINAL\Project\Results\plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

# 1. Load Speech Model and Extract Sub-Models
print("Loading Speech Model...")
speech_model = load_model(SPEECH_MODEL_PATH)
print("Speech Model Loaded!")

# Temporal Modelling Feature Extractor (BiLSTM layer - layer index 2)
# Bidirectional layer is the 3rd layer (index 2) in the sequential model
speech_temporal_extractor = Model(
    inputs=speech_model.inputs,
    outputs=speech_model.layers[2].output
)

# Penultimate Dense Feature Extractor (Dense 64 - layer index 4 or -2)
speech_penultimate_extractor = Model(
    inputs=speech_model.inputs,
    outputs=speech_model.layers[-2].output
)
print("Speech Feature Extractors Ready!")

# 2. Load BERT
print("Loading BERT Tokenizer and Model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()
print("BERT Models Ready!")

# 3. Process TESS Dataset
print("Scanning TESS Dataset...")
X_speech_raw = []
texts = []
labels = []
emotion_names = []

emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
emotion_to_idx = {emo: i for i, emo in enumerate(emotions)}

# Iterate through dataset folders
folders = os.listdir(DATASET_PATH)
for folder in folders:
    folder_path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(folder_path):
        continue
    
    # Skip nested datasets folder if it exists
    if folder.lower() == "tess toronto emotional speech set data":
        continue
        
    emotion = folder.split("_")[-1].lower()
    if emotion == "surprised":
        emotion = "surprise"
        
    if emotion not in emotion_to_idx:
        continue
        
    print(f"Processing folder: {folder} -> mapped to '{emotion}'")
    
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    
    # Process files
    for file_name in tqdm(file_list, desc=f"Loading {emotion}"):
        file_path = os.path.join(folder_path, file_name)
        try:
            # Preprocessing Speech: load and trim silence (resample to 16000Hz)
            audio, sr = librosa.load(file_path, sr=16000)
            
            # Trim silence to satisfy Preprocessing requirement
            audio, _ = librosa.effects.trim(audio)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc = mfcc.T
            
            # Extract word from filename
            parts = file_name.split("_")
            word = parts[1] if len(parts) >= 2 else "word"
            phrase = f"Say the word {word}."
            
            X_speech_raw.append(mfcc)
            texts.append(phrase)
            labels.append(emotion_to_idx[emotion])
            emotion_names.append(emotion)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

print(f"\nSuccessfully loaded {len(X_speech_raw)} samples!")

# Convert labels to numpy arrays
labels = np.array(labels)

# Speech Preprocessing: Pad sequences to max length of 94
print("Padding Speech Sequences...")
X_speech_padded = pad_sequences(
    X_speech_raw,
    maxlen=94,
    padding='post',
    dtype='float32'
)
print("Speech shapes:", X_speech_padded.shape)

# 4. Extract Embeddings
print("Extracting Speech Temporal (BiLSTM) Embeddings...")
temporal_embeddings = speech_temporal_extractor.predict(X_speech_padded, batch_size=64)
print("Temporal Embeddings shape:", temporal_embeddings.shape)

print("Extracting Speech Penultimate (Dense 64) Embeddings...")
speech_pen_embeddings = speech_penultimate_extractor.predict(X_speech_padded, batch_size=64)
print("Speech Penultimate Embeddings shape:", speech_pen_embeddings.shape)

print("Extracting Text Contextual (BERT Pooler) Embeddings...")
text_embeddings_list = []
batch_size = 64
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
print("Text Contextual Embeddings shape:", text_embeddings.shape)

# Fusion Block representation is the concatenation of speech penultimate and text embeddings
fusion_embeddings = np.concatenate([speech_pen_embeddings, text_embeddings], axis=1)
print("Fusion Embeddings shape:", fusion_embeddings.shape)

# 5. Run t-SNE Dimensionality Reduction and Generate Plots
def plot_tsne(features, title, save_filename):
    print(f"Running t-SNE for '{title}'...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    colors = ['#EF4444', '#10B981', '#3B82F6', '#F59E0B', '#6B7280', '#8B5CF6', '#EC4899']
    
    for idx, emo in enumerate(emotions):
        indices = np.where(labels == idx)
        plt.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            label=emo.capitalize(),
            alpha=0.7,
            edgecolors='w',
            color=colors[idx],
            s=40
        )
        
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, save_filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot to {save_path}")

# Plot Temporal Modelling (Speech BiLSTM)
plot_tsne(
    temporal_embeddings,
    "Temporal Modelling Block: Speech (BiLSTM) Embeddings",
    "temporal_clusters.png"
)

# Plot Contextual Modelling (Text BERT)
plot_tsne(
    text_embeddings,
    "Contextual Modelling Block: Text (BERT) Embeddings",
    "contextual_clusters.png"
)

# Plot Fusion Block (Speech Dense + Text BERT)
plot_tsne(
    fusion_embeddings,
    "Fusion Block: Multimodal Joint Embeddings",
    "fusion_clusters.png"
)

print("\nAll cluster visualizations generated successfully!")
