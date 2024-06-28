# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:47:07 2024

@author: Gmaina
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the trained model
loaded_model = pickle.load(open(r"C:\Users\gmaina\trained_model.sav", 'rb'))

#creating a function for prediction
def loanprediction(input_data):
    # Define the scaler and fit it on your training data (or appropriate data)
    #scaler = StandardScaler()
    # Example training data (replace with your actual training data)
    ##X_train = np.array([[1, 1000, 1000, 7, 3]])
    #scaler.fit(X_train)

    # Define your input data
    input_data_numpy_arry = np.array(input_data)
    
    input_data_reshaped = input_data_numpy_arry.reshape(1,-1)

    # Scale the input data
    #scaler.fit(input_data)
    #input_data_scaled = scaler.transform(input_data)

    # Make a prediction
    prediction = loaded_model.predict(input_data_reshaped)

    # Print the prediction result
    print(prediction)

    if prediction[0] == 0:
        return "Likely not to default"
    else:
        return "Likely to default"
    
def main():
    #giving a title
    st.title("Credit Predicition Web App")
    
    #getting the input data from the user
    Customer_ID = st.text_input("What is the customer ID")
    Value = st.text_input("The monetary value of the transaction")
    Amount = st.text_input("Value of the transaction including any charges")
    Product_id = st.text_input("What item is being bought")
    Investor_id = st.text_input("What is the loan issuer or network owner")
    
    #code for predicition
    
    predicta = "" 
    
    # creating button for prediction
    if st.button("Loan Prediction Results"):
        predicta = loanprediction([Customer_ID,Value,Amount,Product_id,Investor_id])
        
    st.success(predicta)
        
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
