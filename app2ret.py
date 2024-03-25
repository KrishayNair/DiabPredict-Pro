import streamlit as st
#import pandas as pd
from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
#import plotly.figure_factory as ff
#from sklearn.metrics import accuracy_score
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
#import seaborn as sns
from streamlit_option_menu import option_menu
import keras
import tensorflow as tf

selected = option_menu(
    menu_title=None,
    options=["Diabetes Retinopathy", "About"],
    icons=["search","book"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)


if selected == "Diabetes Retinopathy":
  # Load your pre-trained model
  model = tf.keras.models.load_model("64x3-CNN.keras")
  #model = keras.layers.TFSMLayer("64x3-CNN.model")

  # Function to preprocess the input image
  def preprocess_image(image):
      image = image.resize((224, 224))
      image = tf.keras.preprocessing.image.img_to_array(image)
      image = np.expand_dims(image, axis=0)
      image = tf.keras.applications.mobilenet.preprocess_input(image)
      return image

  # Function to make predictions
  def predict_diabetes_retinopathy(image):
      preprocessed_image = preprocess_image(image)
      prediction = model.predict(preprocessed_image)
      return prediction
  

# Streamlit App
  def main():
      st.title("Diabetes Retinopathy Prediction System")

      uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

      if uploaded_image is not None:
          image = Image.open(uploaded_image)
          st.image(image, caption="Uploaded Image", use_column_width=True)

          if st.button("Predict"):
              prediction = predict_diabetes_retinopathy(image)
              st.subheader("Prediction Result:")
              st.write(f"Probability of Diabetic Retinopathy: {prediction[0][0]:.2%}")

  if __name__ == "__main__":
      main()