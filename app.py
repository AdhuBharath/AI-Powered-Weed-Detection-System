import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# App title
st.title("AI-Powered Weed Detection System for Sustainable Agriculture")
st.write("Upload your image, System will predict whether it is crop or weed ")

# Load the trained model
@st.cache_resource
def load_model():
    model_path = r"D:\data science\Crop and weed detection project file\crop_weed_detection_model"
    return tf.keras.models.load_model(model_path)

model = load_model()

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    image = image.resize((150, 150))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array /= 255.0  # Normalize

    # Make prediction
    prediction = model.predict(image_array)[0][0]

    # Show result
    label = "Weed" if prediction >= 0.99 else "Crop"
    confidence = (1 - prediction) * 100 if label == "Crop" else prediction * 100

    st.markdown(f"### Prediction: Model says its a **{label}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
    st.markdown( f"pre - {prediction}")