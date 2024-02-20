import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from PIL import Image

model = pickle.load(open("model.sav", "rb"))

st.title("Student Admission Predictions")
st.sidebar.header("Student Data")
image = Image.open("sa.jpg")
st.image(image, "")


def user_report():
    GRE = st.sidebar.slider("GRE Score", 0, 340, 250, step=1)
    TOEFL = st.sidebar.slider("TOEFL Score", 0, 120, 100, step=1 )
    University = st.sidebar.slider("University Rating", 1, 5, 4)
    SOP = st.sidebar.slider("SOP", 0.0, 5.0, 4.0, step=0.5)
    LOR = st.sidebar.slider("LOR", 0.0, 5.0, 4.0, step=0.5)
    CGPA = st.sidebar.slider("CGPA", 0.0, 10.0, 9.0, step=0.1)
    Research = st.sidebar.slider("Research", 0, 1, 1)

    user_report_data = {
        "GRE Score": GRE,
        "TOEFL Score": TOEFL,
        "University Rating": University,
        "SOP": SOP,
        "LOR ": LOR,
        "CGPA": CGPA,
        "Research": Research

    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


user_data = user_report()
st.header("Student Data")
st.write(user_data)

Chance_of_Admit = model.predict(user_data)
st.subheader("Probability of Admission to IVY League Universities")
st.subheader(str(np.round(Chance_of_Admit[0], 2)))
