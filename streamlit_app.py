import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os
import random


st.image('./BAIKElogo.png')

st.write("""
# Dog breed identifier
""")

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:

    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)
    img = img.resize((224,224))
    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Check the type of img_array:
    # Should output: <class 'numpy.ndarray'>
    st.write(type(img_array))

    model = ResNet50(weights='imagenet')
    #img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img_preprocessed)
    st.write(f"Ik denk dat ik een {decode_predictions(prediction, top=3)[0][0][1]}. Met {round(decode_predictions(prediction, top=3)[0][0][2]*100)}% zekerheid.")

