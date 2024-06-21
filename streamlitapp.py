import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import tk
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

gre_score = st.sidebar.slider("GRE score", 1, 10, 5)
toefl_score = st.sidebar.slide('TOEFL score', 1, 10, 5)
uni_rating = st.sidebar.slide('Rating of the University you are applying to', 1,10,5)
sop = st.sidebar.slide('Strength of your Statement of Purpose',1,10,5)
lor = st.sidebar.slide('Strength of your Letter of Recommendation', 1,10,5)
cgpa = st.sidebar.slide('CGPA score',1, 10, 5)
ra = st.selectbox('Resesarch Experirnce (0 for No and 1 for Yes)', 1,10,5)

# PREDICTION
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

st.markdown(prediction)



