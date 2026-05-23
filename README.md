# Multimodal Emotion Recognition System 🎭🤖

A deep learning based Multimodal Emotion Recognition System that combines Speech Emotion Recognition, Text Emotion Recognition, and Multimodal Fusion Learning. The project uses CNN + BiLSTM for speech processing and BERT Transformers for text understanding, then combines both modalities using a fusion network for final emotion prediction.

---

# 🚀 Features

## 🎤 Speech Emotion Recognition
- Audio preprocessing using MFCC
- Deep learning architecture using CNN + BiLSTM
- Predicts emotions directly from speech audio

## 💬 Text Emotion Recognition
- Uses BERT (bert-base-uncased)
- Fine-tuned for emotion classification
- Predicts emotions from text input

## 🔥 Multimodal Fusion
- Combines speech embeddings and text embeddings
- Uses fusion neural network for final emotion prediction

---

# 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| TensorFlow / Keras | Deep learning |
| PyTorch | BERT inference |
| Transformers | NLP models |
| Librosa | Audio processing |
| MFCC | Audio feature extraction |
| CNN | Local speech feature learning |
| BiLSTM | Temporal speech understanding |
| BERT | Text understanding |
| NumPy | Numerical computing |
| Scikit-learn | Data preprocessing |

---

# 📂 Project Structure

Project/
│
├── datasets/
│
├── models/
│   │
│   ├── speech_pipeline/
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── speech_emotion_model.h5
│   │
│   ├── text_pipeline/
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── bert_emotion_model/
│   │
│   ├── fusion_pipeline/
│       ├── train.py
│       ├── test.py
│       ├── fusion_model.keras
│
├── Results/
│   ├── accuracy_table.md
│   ├── plots/
│
├── README.md
│
├── requirements.txt

---

# 🎤 Speech Pipeline Architecture

Audio File
↓
MFCC Extraction
↓
CNN Layer
↓
BiLSTM Layer
↓
Dense Layer
↓
Softmax Emotion Prediction

---

# 💬 Text Pipeline Architecture

Input Text
↓
BERT Tokenizer
↓
BERT Model
↓
Classification Head
↓
Emotion Prediction

---

# 🔥 Fusion Architecture

Speech Embeddings
                    \
                     → Fusion Network → Emotion
                    /
Text Embeddings

---

# 📊 Results

## 🎤 Speech Emotion Recognition
- Achieved near-perfect accuracy on the TESS dataset
- Successfully predicts:
  - Angry
  - Happy
  - Sad
  - Fear
  - Neutral
  - Disgust
  - Surprise

## 💬 Text Emotion Recognition
- Successfully fine-tuned BERT for emotion classification
- Predicts emotions from natural language text

## 🔥 Fusion Model
- Successfully combines speech and text embeddings
- Demonstrates multimodal deep learning pipeline

---

# 📦 Installation

## Clone Repository

git clone <your-github-repo-link>

cd Project

---

## Install Dependencies

pip install -r requirements.txt

---

# ▶️ Running The Project

# 🎤 Speech Pipeline

## Train

cd models/speech_pipeline

python train.py

## Test

python test.py

---

# 💬 Text Pipeline

## Train

cd ../text_pipeline

python train.py

## Test

python test.py

---

# 🔥 Fusion Pipeline

## Train

cd ../fusion_pipeline

python train.py

## Test

python test.py

---

# 📈 Future Improvements

- Real-time emotion recognition
- Better multimodal fusion strategies
- Larger datasets
- Attention mechanisms
- Transformer-based speech models
- Real-world deployment using Flask/FastAPI

---

# 📚 Dataset

This project uses the TESS Dataset (Toronto Emotional Speech Set), which contains emotional speech samples for multiple emotion classes.

---

# 👨‍💻 Author

Dhanush

AI & Deep Learning Enthusiast 🚀

---

# ⭐ Project Highlights

✅ Speech Emotion Recognition  
✅ NLP Emotion Recognition  
✅ CNN + BiLSTM  
✅ BERT Transformers  
✅ Multimodal Fusion AI  
✅ Deep Learning Project  
✅ End-to-End AI Pipeline