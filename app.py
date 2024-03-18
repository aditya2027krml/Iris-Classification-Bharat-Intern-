#Deploying our model as a web application

# importing libraries

import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np

st.text("AUTHOR : <ADITYA KUMAR>")
st.text("TASK 2")


st.header('IRIS CLASSIFICATION')
flower_names = ['iris-setosa', 'iris-versicolour', 'iris-virginica']

model = load_model('Iris_Classification.keras')

# Defining the function to classify images

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of '+ str(np.max(result)*100)
    return outcome


# Defining the function to classify images

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width = 200)

    st.markdown(classify_images(uploaded_file))

   
st.success("Project Done")

