from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
import torch
from tensorflow.keras.models import load_model, Model
from transformers import BertTokenizer, BertForSequenceClassification, BertModel


app = Flask(__name__)

# --- PIPELINE INTEGRATION PLACEHOLDERS ---
# Import your specific loading functions from your pipeline scripts here.
# --- CONFIGURATION PATHS ---
SPEECH_MODEL_PATH = os.path.join("models", "speech_pipeline", "speech_emotion_model.h5")
TEXT_MODEL_PATH = os.path.join("models", "text_pipeline", "bert_emotion_model")
FUSION_MODEL_PATH = os.path.join("models", "fusion_pipeline", "fusion_model.keras")

# --- MODEL INITIALIZATION & LOADING (Loaded once globally on startup) ---
speech_model = load_model(SPEECH_MODEL_PATH)
speech_feature_extractor = Model(inputs=speech_model.inputs, outputs=speech_model.layers[-2].output)

text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_model = BertForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
text_model.eval()

bert_base_model = BertModel.from_pretrained("bert-base-uncased")
bert_base_model.eval()

fusion_model = load_model(FUSION_MODEL_PATH)

# --- LABELS MAPPINGS ---
SPEECH_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
TEXT_EMOTIONS = ['angry', 'fear', 'happy', 'sad']
FUSION_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- PIPELINE INFERENCE HELPERS ---
def preprocess_audio(audio_filepath):
    audio, sr = librosa.load(audio_filepath, sr=16000)
    audio, _ = librosa.effects.trim(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T
    max_len = 94
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len]
    return np.expand_dims(mfcc, axis=0)

def run_text_inference(text_data):
    inputs = text_tokenizer(text_data, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = text_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return TEXT_EMOTIONS[prediction]

def run_speech_inference(audio_filepath):
    mfcc_input = preprocess_audio(audio_filepath)
    prediction = speech_model.predict(mfcc_input)
    return SPEECH_EMOTIONS[np.argmax(prediction[0])]



# --- SERVER ROUTES ---
@app.route('/')
def index():
    # Renders the HTML page inside the templates folder
    return render_template('index.html')

@app.route('/predict_text', methods=['POST'])
def predict_text():
    data = request.get_json()
    user_text = data.get('text', '')
    
    # Run text inference
    predicted_emotion = run_text_inference(user_text)
    
    return jsonify({'emotion': predicted_emotion})

@app.route('/predict_speech', methods=['POST'])
def predict_speech():
    if 'audio' not in request.files:
        return jsonify({'emotion': 'No audio file received'}), 400
        
    audio_file = request.files['audio']
    
    # Save audio temporarily onto disk for processing
    temp_path = os.path.join(os.path.dirname(__file__), "temp_voice_note.wav")

    audio_file.save(temp_path)
    
    try:
        # Run speech inference
        predicted_emotion = run_speech_inference(temp_path)
    finally:
        # Cleanup audio track immediately after testing file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return jsonify({'emotion': predicted_emotion})

@app.route('/predict_fusion', methods=['POST'])
def predict_fusion():
    if 'audio' not in request.files or 'text' not in request.form:
        return jsonify({'error': 'Missing audio or text description'}), 400
        
    audio_file = request.files['audio']
    user_text = request.form['text']
    temp_path = os.path.join(os.path.dirname(__file__), "temp_voice_note_fusion.wav")
    audio_file.save(temp_path)
    
    try:
        # Extract Speech feature vector (1, 64)
        mfcc_input = preprocess_audio(temp_path)
        speech_embedding = speech_feature_extractor.predict(mfcc_input)
        
        # Extract Text embedding vector (1, 768)
        inputs = text_tokenizer(user_text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = bert_base_model(**inputs)
        text_embedding = outputs.pooler_output.numpy()
        
        # Concatenate Features (1, 832)
        fusion_input = np.concatenate([speech_embedding, text_embedding], axis=1)
        
        # Predict
        prediction = fusion_model.predict(fusion_input)
        predicted_class = np.argmax(prediction[0])
        
        return jsonify({
            'emotion': FUSION_EMOTIONS[predicted_class],
            'confidence': float(prediction[0][predicted_class])
        })
    except Exception as e:
         return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    # Starts server debug interface locally
    app.run(debug=True, port=5000)