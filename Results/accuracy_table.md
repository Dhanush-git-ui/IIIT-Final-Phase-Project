# 📊 Model Accuracy & Architecture Comparison

Below is the detailed comparison of the three pipeline variants evaluated on the Toronto Emotional Speech Set (TESS) and emotional cue semantic tasks, detailing the implementation of each functional block:

| Model Variant | Input Modality | Preprocessing Block | Feature Extraction Block | Temporal/Contextual Modelling Block | Classifier Block | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Speech-Only Pipeline** | Speech (WAV) | • Resampled to 16kHz mono<br>• Silence trimmed (Librosa)<br>• Padded to 94 frames | 40-dimensional MFCCs<br>*(time_steps × features)* | Conv1D (Spatial) + Bidirectional LSTM (BiLSTM)<br>*(Sequence learning)* | Dense (64) → Dense (7, Softmax) | **99.00%** |
| **Text-Only Pipeline** | Text | • Cleaned text<br>• Tokenized (BERT Tokenizer) | BERT Embeddings<br>*(tokens × features)* | Pretrained BERT Transformer (`bert-base-uncased`) | `BertForSequenceClassification` head | **100.00%** |
| **Multimodal Fusion Pipeline** | Speech & Text | Combined Speech & Text preprocessing | Concatenated Speech Dense (64d) & BERT pooler (768d) | Joint multimodal feature vector (832d) | Fully Connected Network (Dense 256 → Dense 128 → Softmax) | **99.46%** |

---

## 💡 Key Architectural Insights

1. **Speech Preprocessing**: Silence trimming using `librosa.effects.trim` ensures that only active vocal segments are passed to the model, eliminating ambient noise and silence.
2. **Text Embeddings**: The text pipeline leverages standard pretrained semantic understandings, while the multimodal fusion pipeline uses the 768-dimensional contextual pooler outputs to augment vocal acoustic cues.
3. **Multimodal Fusion Benefits**: Joint feature concatenation (832 dimensions) helps the model resolve ambiguous predictions (e.g., matching a high-pitch voice indicating excitement or anger with a calm semantic statement).