import tensorflow as tf
import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import io

def process_audio(uploaded_file):
    audio, sr = librosa.load(uploaded_file, sr=None)
    return audio, sr

# Function to plot the Mel spectrogram
def plot_melspectrogram(audio, sr):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

    # Plot the Mel spectrogram
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis=None, y_axis=None)
    #librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    #plt.colorbar(format='%+2.0f dB')
    st.pyplot(plt, bbox_inches='tight', pad_inches=0)

# Function to preprocess audio and predict
def predict_audio(model, audio, sr):
    # Convert audio to Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

    # Resize and add extra dimensions for the model input
    mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)  # Add channel dimension
    mel_spec_db = np.expand_dims(mel_spec_db, axis=0)   # Add batch dimension

    # Predict using the model
    prediction = model.predict(mel_spec_db)
    predicted_class = np.argmax(prediction, axis=1)

    # Map predicted class to label
    labels = ["healthy", "covid"]  # Adjust labels based on your model's classes
    predicted_label = labels[predicted_class[0]]

    return predicted_label

def app():

    st.title("Cough Covid-19 Screener")

    # Load the model once at the start
    model = tf.keras.models.load_model('model.keras')
    
    # Add a file uploader for the audio file
    uploaded_file = st.file_uploader("Upload a Cough audio file", type=["wav", "mp3"])


    if uploaded_file is not None:
        st.audio(uploaded_file)
        # Process the uploaded audio file
        audio, sr = process_audio(uploaded_file)

        predicted_label = predict_audio(model, audio, sr)
        if predicted_label == "covid":
            st.title("Prediction: :red[COVID-19]")
        else:
            st.title("Prediction: :green[HEALTHY]")
        
        # Display the Mel spectrogram
        st.subheader("Mel Spectrogram")
        plot_melspectrogram(audio, sr)

        # Predict the class of the uploaded audio
        # Display the predicted label

# Run the Streamlit app
if __name__ == "__main__":
    app()
