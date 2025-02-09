# Import modules
import streamlit as st 
import tensorflow as tf 
import numpy as np 
import gdown
import os

# File information
url = "https://drive.google.com/uc?id=1JFBFGybOctCYqau4f1s9U9Jt2W7PeCDa"
file_id = "1JFBFGybOctCYqau4f1s9U9Jt2W7PeCDa"
model_path = "potato_leaf_disease_detection_model.keras"

# If path not exists then download
if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)
    st.warning("Model has been downloaded...")

# Perform prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Set up the application
st.sidebar.title('Potato Plant Leaf Disease Detection System')
app_mode = st.sidebar.selectbox('select page', ['Home', 'Disease Recognition'])

# Import image
from PIL import Image
img = Image.open('Plant_Image.jpeg')
st.image(img)

# If it is home page
if (app_mode == 'HOME'):
    st.markdown("<h1 style='text-align: center;'>Potato Plant Leaf Disease Detection System", unsafe_allow_html=True)

# If it is disease recognition page
if (app_mode == 'Disease Recognition'):
    st.header('Potato Plant Leaf Disease Detection System')

# To upload an image
test_image = st.file_uploader("Choose an image: ")
if (st.button("Show Image")):
    st.image(test_image, width=4, use_container_width=True)

# Display the prediction on clicking button
if (st.button('Predict')):
    st.snow()
    st.write('Our Prediction')
    result_index = model_prediction(test_image)
    class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    class_names_dict = {
        'Potato___Early_blight': 'Potato Early Blight',
        'Potato___Late_blight': 'Potato Late Blight',
        'Potato___healthy': 'Healthy Potato'
    }
    st.success(f"Model is predicting it is a {class_names_dict[class_names[result_index]]}.")

