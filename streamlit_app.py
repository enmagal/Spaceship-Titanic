import streamlit as st

import pandas as pd

df_train = pd.read_csv("./spaceship-titanic-data/train.csv")



add_sidebar = st.sidebar.selectbox('Part of data discovery', ('Train Dataset','Missing Values'))

st.title('🧑‍🚀 Spaceship Titanic')

if add_sidebar == 'Train Dataset':
    st.header('Train Dataset')
    st.write(df_train)

elif add_sidebar == 'Missing Values':
    st.header('Missing Values')