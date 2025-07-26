import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load ANN model and encoders
model = tf.keras.models.load_model('ann_model.h5')

with open('lbl_enc.pkl', 'rb') as file:
    lbl_enc = pickle.load(file) 

with open('ohe.pkl', 'rb') as file:
    ohe = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# UI
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', ohe.categories_[0])  # Use categories from loaded ohe
gender = st.selectbox('Gender', lbl_enc.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [lbl_enc.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode Geography
geo_enc = ohe.transform([[geography]]).toarray()
geo_enc_df = pd.DataFrame(geo_enc, columns=ohe.get_feature_names_out(['Geography']))

# Combine with input
input_data = pd.concat([input_data.reset_index(drop=True), geo_enc_df], axis=1)

# Scale input
input_data_scaled = scaler.transform(input_data)

# Predict
predictions = model.predict(input_data_scaled)[0][0]

st.write(f'Churn Probability: {predictions:.2f}')
if predictions > 0.5:
    st.write('🚨 The customer is likely to churn.')
else:
    st.write('✅ The customer is not likely to churn.')
