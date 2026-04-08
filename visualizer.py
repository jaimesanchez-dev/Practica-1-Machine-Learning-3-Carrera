import pandas as pd
import streamlit as st

import pickle as pkl

file = "./dataset/bank_10.pkl"

with open(file, 'rb') as fd:
    df = pkl.load(fd)
    print(df)

# Extract the number of variables and instances
df_shape = df.shape

# Create a chart
st.subheader("Número de instancias y variables")
st.table(df_shape)
