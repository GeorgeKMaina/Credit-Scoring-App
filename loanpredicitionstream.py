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
from sklearn.preprocessing import LabelEncoder

# Load the trained model
loaded_model = pickle.load(open(r"C:\Users\gmaina\Downloads\trained_model.sav", 'rb'))
product_encoder = pickle.load(open(r"C:\Users\gmaina\Downloads\product_encoder.sav", 'rb'))
investor_encoder = pickle.load(open(r"C:\Users\gmaina\Downloads\investor_encoder.sav", 'rb'))

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
    
    #select boxes for product
    product=('airtime', 
             'data_bundles', 
             'financial_services',
             'movies',
             'retail',
             'tv',
             'utility_bill'
             )
    
    investor=('InvestorId_1',
              'InvestorId_2',
              'InvestorId_3',
              'NA'
              )
    
    # Initialize and fit the LabelEncoder for products and investors
    product_encoder = LabelEncoder()
    product_encoder.fit(product)
    investor_encoder= LabelEncoder()
    investor_encoder.fit(investor)
    
    
    
    
    #getting the input data from the user
    Customer_ID = st.text_input("What is the customer ID")
    Value = st.slider("The monetary value of the transaction",min_value=0, max_value=10000)
    if Value == 0:
        Amount = st.slider("Value of the transaction including any charges", min_value=0, max_value=10000)
    else:
        Amount = st.slider("Value of the transaction including any charges", min_value=Value, max_value=int(Value + (0.2 * Value)))
        
    Product_id = st.selectbox("What item is being bought",product)
    Investor_id = st.selectbox("What is the loan issuer or network owner",investor)
    
    # Encode categorical variables
    Product_id_encoded = product_encoder.transform([Product_id])[0]
    Investor_id_encoded = investor_encoder.transform([Investor_id])[0]

    # Convert input data to appropriate types
    input_data = [int(Customer_ID), float(Value), float(Amount), Product_id_encoded, Investor_id_encoded]

    
    #code for predicition
    
    predicta = "" 
    
    # creating button for prediction
    if st.button("Loan Prediction Results"):
        predicta = loanprediction(input_data)
        
    st.success(predicta)
        
if __name__ == '__main__':
    main()
