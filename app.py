import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Load the trained LSTM model
model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40, 1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])
model.load_weights('modelEA.h5')  # Load model weights

# Function to extract MFCC features from audio file
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Function to preprocess input audio for model prediction
def preprocess_audio(audio_file):
    mfcc = extract_mfcc(audio_file)
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
    mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension
    return mfcc

# Function to predict emotion from audio file
def predict_emotion(audio_file):
    processed_audio = preprocess_audio(audio_file)
    prediction = model.predict(processed_audio)
    emotion_label = np.argmax(prediction)
    return emotion_label

# Streamlit UI
st.title("Emotion Detection from Audio")

# File upload widget
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# Prediction button
if audio_file is not None:
    # Perform prediction when audio file is uploaded
    st.audio(audio_file, format='audio/wav')
    emotion_label = predict_emotion(audio_file)
    st.write("Predicted Emotion:", emotion_label)
