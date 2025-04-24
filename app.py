import os
import keras
import numpy as np
from keras.models import load_model
import streamlit as st
import tensorflow as tf

# Set Streamlit header
st.header("Image Classification CNN Model")

# Flower class names
flower_name = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load the pre-trained model
model = load_model('Flower_recognition_Model.h5')

# Classification function
def classify_images(image_path):
    input_images = tf.keras.utils.load_img(image_path, target_size=(180, 180))  
    input_images_array = tf.keras.utils.img_to_array(input_images)
    input_images_exp_dim = tf.expand_dims(input_images_array, 0)

    predictions = model.predict(input_images_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + flower_name[np.argmax(result)] + ' with a score of ' + str(np.max(result) * 100)
    return outcome

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_dir = "tempDir"
    os.makedirs(temp_dir, exist_ok=True)
    image_path = os.path.join(temp_dir, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(uploaded_file, width=200)

    # Show prediction result
    result = classify_images(image_path)
    st.markdown(result)
