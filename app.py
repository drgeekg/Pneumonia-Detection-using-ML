import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Load pre-trained model
model = load_model('trained.h5')

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((300, 300))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict Pneumonia
def predict_pneumonia(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    confidence = prediction[0][0]
    result = 'Pneumonia' if confidence > 0.5 else 'Normal'
    return result, confidence

# Streamlit app
st.title("Pneumonia Detection Using Convolutional Neural Networks")

st.write("Upload a chest X-ray image to detect Pneumonia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict and display result
    result, confidence = predict_pneumonia(img)
    st.write(f'The model predicts: {result}')
    st.write(f'Confidence level: {confidence:.2f}')
