import streamlit as st
import pandas as pd
import numpy as np
from PIL import ImageTk, Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os
import random


map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)


st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

st.write(""" Hello World
# Very nice
Nice *test* how hard is this""")

st.image('./BAIKElogo.png')


st.write(""" # take picture """)
picture = st.camera_input("Take a picture")

if picture:
     st.image(picture)
     # define ResNet50 model
    model = ResNet50(weights='imagenet')
    img = image.load_img(picture, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img_preprocessed)
    st.write(f"Ik denk dat ik een {decode_predictions(prediction, top=3)[0][0][1]}. Met {round(decode_predictions(prediction, top=3)[0][0][2]*100)}% zekerheid.")
