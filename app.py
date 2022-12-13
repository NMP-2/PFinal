import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import plotly as pl
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


    
st.markdown ("## Predicting LinkedIn Users")

st.markdown("#### Please provide the following information to allow my algorithm to predict if you are a linked in user or not.")

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
   bi = np.where((x == 1), 1, 0)
   return bi

ss = pd.DataFrame({
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"]> 8,np.nan, s["educ2"]),
    "parent":np.where(s["par"]>2, np.nan, s["par"]),
    "parent":np.where(s["par"] ==2, 0, s["par"]),
    "married":np.where(s["marital"]>1, 0, 1),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s["age"] > 97, np.nan, s["age"]),    
    "sm_li":clean_sm(s["web1h"])
    })


ss =ss.dropna()

y = ss["sm_li"]
x = ss[["income", "education", "parent", "married", "female", "age"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2,random_state=307)


lr = LogisticRegression(class_weight="balanced")
lr.fit(x_train, y_train)

income = st.selectbox("Please Select your Income",
        options = ["<$10k",
            "$10k - $20k",
            "$20k - $30k",
            "$30k - $40k",
            "$40k - $50k",
            "$50k - $75k",
            "$75k - $100k",
            "100k - $150k",
            ">$150k"])
if income == "<$10k":
    income = 1
elif income == "$10k - $20k":
    income = 2
elif income == "$20k - $30k":
    income = 3
elif income == "$30k - $40k":
    income = 4
elif income == "$40k - $50k":
    income = 5
elif income == "$50k - $75k":
    income = 6
elif income == "$75k - $100k":
    income = 7
elif income == "100k - $150k":
    income = 8
else: 
    income = 9


education = st.selectbox("Please Select your Education Level",
        options = ["Less than High school",
            "Did not finish High school",
            "Highschool graduate or GED",
            "Some college",
            "Associate's Degree",
            "Bachelor's Degree",
            "Some Postgrad or Professional schooling",
            "Post Graduates or Professional Degree i.e. MA, PhD, MD, JD, etc."])
if education == "Post Graduates or Professional Degree i.e. MA, PhD, MD, JD, etc.":
    education = 8
elif education == "Did not finish High school":
    education = 2
elif education == "Highschool graduate or GED":
    education = 3
elif education == "Some college":
    education = 4
elif education == "Associate's Degree":
    education = 5
elif education == "Bachelor's Degree":
    education = 6
elif education == "Some Postgrad or Professional schooling":
    education = 7
else: 
    education = 1

parent = st.selectbox("Are you a parent, with a child living at home?",
            options = ["Yes",
                        "No"])
if parent == "Yes":
    parent = 1
else:
    parent = 0

married = st.selectbox("Are you currently married?",
            options = ["Yes",
                        "No"])
if married == "Yes":
    married = 1
else:
    married = 0

female = st.selectbox("What Gender do you Identify as?",
            options = ["Male",
                        "Female",
                        "Non-Binary",
                        "Other"])
if female == "Female":
    female = 1
else:
    female = 0

age = st.slider("Please enter your age:", min_value = 13, max_value = 110)


if st.button("PREDICT"):
    person = [income, education, parent, married, female, age]

    prediction = lr.predict([person])
    prediction_1 = prediction
    if prediction == 1:
         prediction_1 = "User"
    else:
         prediction_1 = "Non-User"

    prob = round((lr.predict_proba([person])[0][1] * 100), 2) 

    if prob >=50:
        f"Based on the regression, I predict you are a LinkedIn user.\
                There is a {prob} percent chance you are a user."
    else: 
        f"Based on the regression, I predict you are not a LinkedIn user.\
                There is a {prob} percent chance you are a user."

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob,
        title = {'text' : f"Prediction: {prediction_1}"},
        gauge = {"axis": {"range": [0,100]},
                "steps": [
                    {"range": [0, 40], "color":"lightgreen"},
                    {"range": [40, 60], "color":"gray"},
                    {"range": [60, 100], "color":"green"}
                ],
                "bar":{"color":"lightblue"}}
    ))
        
    st.plotly_chart(fig)