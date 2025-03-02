
import os
import torch
from money_laundering import TabularModel
import joblib
import streamlit as st
import pandas as pd
import numpy as np


model = torch.load("C://Users/honer/Downloads/SCAIq2/ML_predictor.pkl", weights_only=False)
model.eval()

st.title("Predicting if a given transaction is fraudulent or legitimate.")

col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input("Amount", step=0.1)

with col2:
    hour = st.number_input("Hour of the day", step=0.1)

with col3:
    payment_currency = st.text_input('Payment currency')

with col1:
    received_currency = st.text_input('Received currency')

with col2:
    sender_country = st.text_input('Sender country')

with col3:
    receiver_country = st.text_input("Receiver country")

with col1:
    payment_type = st.text_input('Payment Type')

cat_cols = ["Payment_currency", "Received_currency", "Sender_bank_location", "Receiver_bank_location", "Payment_type"]
cont_cols = ["Amount", "Hour"]

if st.button('Check if fraudulent: '):
    user_conts = [[amount, hour]]
    user_cats = [[payment_currency, received_currency, sender_country, receiver_country, payment_type]]

    emb_szs = [(13, 7), (13, 7), (18, 9), (18, 9), (7, 4)]
    cat_vals = pd.DataFrame(data=user_cats, columns= cat_cols)
    cont_vals = pd.DataFrame(data=user_conts, columns= cont_cols)
    for cat in cat_cols:
        cat_vals[cat] = cat_vals[cat].astype('category')
    cats = np.stack([cat_vals[col].cat.codes.values for col in cat_cols], 1)
    cats = torch.tensor(cats, dtype=torch.int64)

    conts = np.stack([cont_vals[col].values for col in cont_cols], 1)
    conts = torch.tensor(conts, dtype=torch.float)

    y_pred = model(cats, conts)
    print(y_pred)
    prob_class_0 = (y_pred[0][0]).detach().numpy()
    prob_class_1 = (y_pred[0][1]).detach().numpy()
    print("prob class 0: ", prob_class_0)
    print("prob class 1: ", prob_class_1)


    diff = prob_class_1 - prob_class_0

    if diff < -1.2:
        st.success("This transaction is legitimate ", icon="✅")
    elif (diff > -1.2) and (diff < 0):
        st.warning("There is a chance this transaction is fraudulent. Keep a close eye", icon="⚠")
    else:
        st.error("This transaction is fraudulent.", icon="🚨")

    #st.success(prediction)
