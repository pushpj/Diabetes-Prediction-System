# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#Loading the model
loaded_model = pickle.load(open('D:/DESKTOP/Diabetes_Prediction_Model/trained_model.sav', 'rb'))

input_data = (6,190,92,0,0,35.5,0.278,66) #The Outcome is 1 so our model should also predict 1 as the right answer

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
  print("Therefore, the person has Diabetes :(")
else:
  print("Therefore, the person does not have Diabetes :)")