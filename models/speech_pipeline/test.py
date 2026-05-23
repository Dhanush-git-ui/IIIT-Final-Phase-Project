import numpy as np
import librosa

from tensorflow.keras.models import load_model



model = load_model("speech_emotion_model.h5")

print("Speech Model Loaded!")



audio_path = r"C:\Users\dhanu\OneDrive\Desktop\IIIT GRAND FINAL\Project\datasets\TESS Toronto emotional speech set data\OAF_angry\OAF_back_angry.wav"


audio, sr = librosa.load(
    audio_path,
    sr=16000
)

print("Sampling Rate:", sr)



mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

# Transpose
mfcc = mfcc.T



max_len = 94

if mfcc.shape[0] < max_len:

    pad_width = max_len - mfcc.shape[0]

    mfcc = np.pad(
        mfcc,
        pad_width=((0, pad_width), (0, 0))
    )

else:

    mfcc = mfcc[:max_len]



mfcc = np.expand_dims(
    mfcc,
    axis=0
)

print("Input Shape:", mfcc.shape)



prediction = model.predict(mfcc)



emotions = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise'
]

predicted_index = np.argmax(prediction[0])

predicted_emotion = emotions[predicted_index]



print("\nPredicted Emotion:")
print(predicted_emotion)