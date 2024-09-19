import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load the trained CNN tumor classification model
model = tf.keras.models.load_model("path_to_your_saved_model/cnn_tumor.h5")

# Function to make predictions
def make_prediction(image, model):
    img = Image.open(image)
    img = img.resize((128, 128))  # Resize to match the model's input size
    img = np.array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.keras.utils.normalize(img, axis=1)  # Normalize the image

    prediction = model.predict(img)
    return "Tumor Detected" if prediction > 0.5 else "No Tumor"

# Streamlit app
st.title("Tumor Classification with CNN")
st.write("Upload a medical image for tumor classification using a CNN model.")

# Image file upload
uploaded_file = st.file_uploader("Upload a medical image (JPG/PNG)", type=["jpg", "png"])

# Button to proceed with tumor classification
if st.button("Classify"):
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make a prediction
        result = make_prediction(uploaded_file, model)

        # Display the result
        st.write(f"Prediction: {result}")
    else:
        st.write("Please upload an image file.")
