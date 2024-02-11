
# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from streamlit_option_menu import option_menu

import tensorflow as tf



df = pd.read_csv("diabetesnew.csv")

selected = option_menu(
    menu_title=None,
    options=["Diabetes Prediction","Diabetes Retinopathy", "About"],
    icons=["search","search","book"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)
if selected == "Diabetes Prediction":
# HEADINGS
  st.title('Diabetes Checkup')
  st.sidebar.header('Patient Data')
  st.subheader('Training Data Stats')
  st.write(df.describe())


  # X AND Y DATA
  x = df.drop(['Outcome'], axis = 1)
  y = df.iloc[:, -1]
  x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


  # FUNCTION
  def user_report():
      pregnancies = st.sidebar.slider('Pregnancies', 0,17,3)
      glucose = st.sidebar.slider('Glucose', 0,200, 120 )
      bp = st.sidebar.slider('Blood Pressure', 0,122, 70 )
      skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
      insulin = st.sidebar.slider('Insulin', 0,846, 79 )
      bmi = st.sidebar.slider('BMI', 0,67, 20 )
      dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
      age = st.sidebar.slider('Age', 21,88, 33 )

      user_report_data = {
          'Pregnancies': pregnancies,
          'Glucose': glucose,
          'BloodPressure': bp,
          'SkinThickness': skinthickness,
          'Insulin': insulin,
          'BMI': bmi,
          'DiabetesPedigreeFunction': dpf,
          'Age': age
      }
      report_data = pd.DataFrame(user_report_data, index=[0])
      return report_data




  # PATIENT DATA
  user_data = user_report()
  st.subheader('Patient Data')
  st.write(user_data)




  # MODEL
  rf  = RandomForestClassifier()
  rf.fit(x_train, y_train)
  user_result = rf.predict(user_data)
  #user_result = loaded_model.predict(user_data)

  

  # VISUALISATIONS
  st.title('Visualised Patient Report')



  # COLOR FUNCTION
  if user_result[0]==0:
    color = 'blue'
  else:
    color = 'red'


  # Age vs Pregnancies
  st.header('Pregnancy count Graph (Others vs Yours)')
  fig_preg = plt.figure()
  ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
  ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['Pregnancies'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,20,2))
  plt.title('0 - Healthy & 1 - Unhealthy')
  st.pyplot(fig_preg)



  # Age vs Glucose
  st.header('Glucose Value Graph (Others vs Yours)')
  fig_glucose = plt.figure()
  ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
  ax4 = sns.scatterplot(x = user_data['Age'], y = user_data['Glucose'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,220,10))
  plt.title('0 - Healthy & 1 - Unhealthy')
  st.pyplot(fig_glucose)



  # Age vs Bp
  st.header('Blood Pressure Value Graph (Others vs Yours)')
  fig_bp = plt.figure()
  ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
  ax6 = sns.scatterplot(x = user_data['Age'], y = user_data['BloodPressure'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,130,10))
  plt.title('0 - Healthy & 1 - Unhealthy')
  st.pyplot(fig_bp)


  # Age vs St
  st.header('Skin Thickness Value Graph (Others vs Yours)')
  fig_st = plt.figure()
  ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
  ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['SkinThickness'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,110,10))
  plt.title('0 - Healthy & 1 - Unhealthy')
  st.pyplot(fig_st)


  # Age vs Insulin
  st.header('Insulin Value Graph (Others vs Yours)')
  fig_i = plt.figure()
  ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
  ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['Insulin'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,900,50))
  plt.title('0 - Healthy & 1 - Unhealthy')
  st.pyplot(fig_i)


  # Age vs BMI
  st.header('BMI Value Graph (Others vs Yours)')
  fig_bmi = plt.figure()
  ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
  ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['BMI'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,70,5))
  plt.title('0 - Healthy & 1 - Unhealthy')
  st.pyplot(fig_bmi)


  # Age vs Dpf
  st.header('DPF Value Graph (Others vs Yours)')
  fig_dpf = plt.figure()
  ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
  ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['DiabetesPedigreeFunction'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,3,0.2))
  plt.title('0 - Healthy & 1 - Unhealthy')
  st.pyplot(fig_dpf)



  # OUTPUT
  st.subheader('Your Report: ')
  output=''
  if user_result[0]==0:
    output = 'You are not Diabetic'
  else:
    output = 'You are Diabetic'
  st.title(output)
  st.subheader('Accuracy: ')
  st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')


if selected == "Diabetes Retinopathy":
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


if selected == "About":
  st.header("What is Diabetes?")
  st.write("Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy.")
  st.write("Your body breaks down most of the food you eat into sugar (glucose) and releases it into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin. Insulin acts like a key to let the blood sugar into your body’s cells for use as energy.")

  st.write("With diabetes, your body doesn't make enough insulin or can’t use it as well as it should. When there isn’t enough insulin or cells stop responding to insulin, too much blood sugar stays in your bloodstream. Over time, that can cause serious health problems, such as heart disease, vision loss, and kidney disease.")

  st.write("There isn’t a cure yet for diabetes, but losing weight, eating healthy food, and being active can really help. Other things you can do to help:")

  st.write("1. Take medicine as prescribed.")
  st.write("2. Get diabetes self-management education and support.")
  st.write("3. Make and keep health care appointments.")
  st.image("D1.jpg")


