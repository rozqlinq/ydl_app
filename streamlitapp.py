import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
matplotlib.use('tkagg')

adm = pd.read_csv('/Users/rozalina.alkeyeva/Desktop/adm_data.csv')

# CHECKING NAN
has_nan = adm.isnull().sum()

# CONVERTING GPA VALUES
new_gpa = []

for i in range (len(adm['CGPA'])):
    new_gpa.append((adm['CGPA'][i]/10)*4)

adm['CGPA']=new_gpa

# TRAINING MODEL
X = adm.drop(["Serial No.", "Chance of Admit "], axis=1)
Y = adm['Chance of Admit ']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

lm=LinearRegression()
lm.fit(X_train, Y_train)

# APP INTERFACE
st.set_page_config(layout='centered')

st.title('University Admissions Predictor')
st.write("""Welcome to University Admissions Predictor app!
            This app predicts your chances of admission to an university
            Please enter the information about your application package required below""")

with st.form(key = 'information', clear_on_submit=True):
    gre_score = st.number_input('Enter your GRE score')
    toefl_score = st.number_input('Enter your TOEFL score')
    uni_rating = st.selectbox('Enter the Rating of the University you are applying to', [1,2,3,4,5])
    sop = st.selectbox('Enter the approximate Strength of your Statement of Purpose', [1,1.5,2,2.5,3,3.5,4,4.5,5])
    lor = st.selectbox('Enter the approximate Strength of your Letter of Recommendation',
                       [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    cgpa = st.number_input('Enter your CGPA score')
    ra = st.selectbox('Do you have Resesarch Experirnce? (0 for No and 1 for Yes)', [0,1])

# PREDICTION
if st.form_submit_button('Predict'):
    data = pd.DataFrame({
        'GRE Score': [gre_score],
        'TOEFL Score': [toefl_score],
        'University Rating': [uni_rating],
        'SOP': [sop],
        'LOR': [lor],
        'CGPA': [cgpa],
        'Research': [ra]
    })

prediction = lm.predict(data)

st.balloons()
st.success(f"Predicted Probability: {prediction[0]:,.2f}",icon="✅")


