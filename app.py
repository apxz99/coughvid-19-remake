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
    return tf.keras.models.load_model('best_model3.keras')

def process_audio(audio_bytes):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    return audio, sr

# Function to plot the Mel spectrogram
def predict_audio(model, audio, sr):
    # แปลงเสียงเป็น melspectrogram
    mels_db = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr), ref=1.0)
    
    # ปรับรูปแบบให้ตรงกับ input ของโมเดล
    mels_db = cv2.resize(mels_db, (128, 32))  # ปรับขนาดตามที่โมเดลคาดหวัง
    mels_db = np.expand_dims(mels_db, axis=-1)  # เพิ่มมิติ channel
    mels_db = np.expand_dims(mels_db, axis=0)   # เพิ่มมิติ batch
    
    # ทำนายผลลัพธ์
    predictions = model.predict(mels_db)
    predicted_class = np.argmax(predictions, axis=1)
    
    # แปลงผลลัพธ์เป็น label
    status_labels = ["healthy", "covid"]  # ปรับชื่อคลาสตามโมเดลที่ฝึก
    predicted_label = status_labels[predicted_class[0]]
    
    return predicted_label

def app():
    model = load_model()
    st.title("Cough Covid-19 Screener")
    st.subheader("วิธีการใช้งาน ")
    st.text(
    """1. เลือกวิธีการอัพโหลดไฟล์เสียง / Choose a method to upload the audio file.
        2. อัพโหลด/บันทึก "เสียงไอ" / Upload or Record a "Cough audio".'
        3. รอการประมวลผล / Receive the prediction result.
        """
        )

    # Option to upload file or record audio
    option = st.radio("1 . Choose a method to upload the audio file :", ("Upload a file", "Record audio"))

    if option == "Upload a file":
        uploaded_file = st.file_uploader("2 . Upload a Cough audio file", type=["wav", "mp3"])
        
        if uploaded_file is not None:
            audio_data, sr = process_audio(uploaded_file.read())
            st.audio(uploaded_file)
            with st.spinner("Analyzing audio..."):
                time.sleep(1)
                predicted_label = predict_audio(model, audio_data, sr)
                if predicted_label == "covid":
                    st.title(f"Prediction: :red[{predicted_label}]")
                else:
                    st.title(f"Prediction: :green[{predicted_label}]")
                st.success("Prediction Success!", icon="✅")


    elif option == "Record audio":
        # Record audio
        rec = st.audio_input("2 . Record a cough")

        if rec is not None:
            audio_data, sr = process_audio(rec.read())
            st.audio(rec)
            with st.spinner("Analyzing audio..."):
                time.sleep(1)
                predicted_label = predict_audio(model, audio_data, sr)
                if predicted_label == "covid":
                    st.title(f"Prediction: :red[{predicted_label}]")
                else:
                    st.title(f"Prediction: :green[{predicted_label}]")

# Run the Streamlit app
if __name__ == "__main__":
    app()
