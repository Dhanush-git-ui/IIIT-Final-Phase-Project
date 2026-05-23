import numpy as np

from tensorflow.keras.models import load_model

# =========================================================
# LOAD FUSION MODEL
# =========================================================

model = load_model(
    "fusion_model.keras"
)

print("Fusion Model Loaded!")

# =========================================================
# CREATE SAMPLE INPUT
# =========================================================

sample_input = np.random.rand(
    1,
    832
)

print("Input Shape:", sample_input.shape)

# =========================================================
# PREDICT
# =========================================================

prediction = model.predict(
    sample_input
)

predicted_class = np.argmax(
    prediction[0]
)

# =========================================================
# LABELS
# =========================================================

emotions = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

predicted_emotion = emotions[
    predicted_class
]

# =========================================================
# OUTPUT
# =========================================================

print("\nPredicted Emotion:")
print(predicted_emotion)