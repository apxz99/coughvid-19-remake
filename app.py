import tensorflow as tf
import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import io
import time

def load_model():
    return tf.keras.models.load_model('200_model.keras')

def process_audio(audio_bytes):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    return audio, sr

# Function to plot the Mel spectrogram
def plot_melspectrogram(audio, sr):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

    # Plot the Mel spectrogram
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format='%+2.0f dB')
    st.pyplot(plt, bbox_inches='tight', pad_inches=0.1)

# Function to preprocess audio and predict
def predict_audio(model, audio, sr):
    # Convert audio to Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

    # Resize to match model's expected input shape
    expected_shape = (128, 32)  # Replace with your model's input shape
    mel_spec_db_resized = cv2.resize(mel_spec_db, expected_shape)

    # Add necessary dimensions for model input
    mel_spec_db_resized = np.expand_dims(mel_spec_db_resized, axis=-1)  # Add channel dimension
    mel_spec_db_resized = np.expand_dims(mel_spec_db_resized, axis=0)   # Add batch dimension

    # Predict using the model
    prediction = model.predict(mel_spec_db_resized)
    predicted_class = np.argmax(prediction, axis=1)

    # Map predicted class to label
    labels = ["healthy", "covid"]  # Adjust labels based on your model's classes
    predicted_label = labels[predicted_class[0]]

    return predicted_label

def app():
    model = load_model()
    st.title("Cough Covid-19 Screener")

    # Option to upload file or record audio
    option = st.radio("Choose how to provide the audio:", ("Upload a file", "Record audio"))

    if option == "Upload a file":
        uploaded_file = st.file_uploader("Upload a Cough audio file", type=["wav", "mp3"])
        
        if uploaded_file is not None:
            audio_data, sr = process_audio(uploaded_file.read())
            st.audio(uploaded_file)
            with st.spinner("Analyzing audio..."):
                predicted_label = predict_audio(model, audio_data, sr)
                time.sleep(1)
                st.title(f"Prediction: {predicted_label}")
            st.subheader("Mel Spectrogram")
            plot_melspectrogram(audio_data, sr)

    elif option == "Record audio":
        # Record audio
        rec = st.audio_input("Record a cough")

        if rec is not None:
            audio_data, sr = process_audio(rec.read())  
            st.audio(rec)
            with st.spinner("Analyzing audio..."):
                predicted_label = predict_audio(model, audio_data, sr)
                time.sleep(1)
                st.title(f"Prediction: {predicted_label}")
            st.subheader("Mel Spectrogram")
            plot_melspectrogram(audio_data, sr)

# Run the Streamlit app
if __name__ == "__main__":
    app()
