import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your pre-trained model
model = tf.keras.models.load_model("64x3-CNN.model")

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
