# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 21:09:48 2022

@author: Pushp jain
"""

import numpy as np
import pickle
import streamlit as st

#Loading the model
loaded_model = pickle.load(open('D:/DESKTOP/Diabetes_Prediction_Model/trained_model.sav', 'rb'))

#creating function for prediction
def diabetes_prediction(input_data):
    #Change the input array to a numpy array
    input_data_np_array = np.asarray(input_data)

    #Reshaping the np array as we're predicting for only one instance
    input_data_reshape = input_data_np_array.reshape(1, -1)

    # #Standardising the data
    # input_standardized_data = scaler.transform(input_data_reshape)

    #Predicting data variable
    input_data_prediction = loaded_model.predict(input_data_reshape) #This predicts the Outcome
    print("Predicted value of input data =", input_data_prediction)

    if(input_data_prediction[0] == 1):
      return 'The person is Diabetic :('
    else:
      return 'The person is not Diabetic :)'
  

def main():
    #Giving title
    st.title('Diabetes Prediction System')
    
    #Getting input from user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Blood Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Blood Insulin Level')
    BMI = st.text_input('Body Mass Index (BMI) Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Enter Age')
    
    #For prediction
    diagnosis = ''
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)
    

if __name__ == '__main__':
    main()
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    