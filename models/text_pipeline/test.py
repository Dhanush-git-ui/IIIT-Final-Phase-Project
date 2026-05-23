import torch

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)



model = BertForSequenceClassification.from_pretrained(
    "bert_emotion_model"
)

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

print("BERT Emotion Model Loaded!")



text = "I am feeling very happy today"



inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True
)



with torch.no_grad():

    outputs = model(**inputs)

prediction = torch.argmax(
    outputs.logits,
    dim=1
)



labels = [
    "angry",
    "fear",
    "happy",
    "sad"
]

predicted_emotion = labels[
    prediction.item()
]


print("\nPredicted Emotion:")
print(predicted_emotion)