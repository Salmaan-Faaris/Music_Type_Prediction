import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# Load the pre-trained model and feature extractor
model_name = "dima806/music_genres_classification"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

# Function to preprocess audio
def preprocess_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    inputs = feature_extractor(audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
    return inputs

# Function to classify music genre
def predict_genre(audio_path):
    inputs = preprocess_audio(audio_path)
    
    # Run inference
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_genre = model.config.id2label[predicted_id]
    
    return predicted_genre

audio_file = "C:\\Users\\salma\\Downloads\\rock-cinematic-161648.wav"  
predicted_genre = predict_genre(audio_file)
print(f"Predicted Genre: {predicted_genre}")
