from flask import Flask, render_template,request
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
import streamlit as slt

model=pickle.load(open("LR_model.pkl",'rb'))

slt.title("Stock Prediction")

data=pd.read_csv('GOOG.csv')
close=data['close']
close=pd.DataFrame(close)
l=close.shape[0]

user_input=slt.text_input("Enter the number of days")

newdata=close
if(user_input!=""):
    t=int(user_input)
    for i in range(l-50,(l-49)+t):
        prev_data=[newdata['close'][i:i+50]]
        predicted_stock=model.predict(prev_data)
        newdata.loc[len(newdata.index)] = [predicted_stock[0]] 

    fig=plt.figure(figsize=(12,8))
    plt.plot(newdata['close'])
    plt.plot(close['close'])
    slt.pyplot(fig)
else:
    # plt.rcParams["figure.figsize"]=(12,8)
    fig=plt.figure(figsize=(12,8))
    plt.plot(close['close'],label="actual")
    plt.legend(loc="upper left")
    slt.pyplot(fig)








