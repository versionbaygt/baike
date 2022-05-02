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

picture = "./BAIKElogo.png"
img = Image.open(picture)
img_array_aux = np.array(img)
# define ResNet50 model
model = ResNet50(weights='imagenet')
img = image.load_img(picture, target_size=(224, 224))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_batch)
prediction = model.predict(img_preprocessed)
print(f"Ik denk dat ik een {decode_predictions(prediction, top=3)[0][0][1]}. Met {round(decode_predictions(prediction, top=3)[0][0][2]*100)}% zekerheid.")
