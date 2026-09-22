
import streamlit as st
import pandas as pd
from pathlib import Path

st.title("Maternal Health Risk Dashboard")

# Load the dataset
data_path = Path(__file__).parent / "Maternal Health Risk Data Set.csv"

df = pd.read_csv(data_path)

# Display the dataset
st.header("1. Dataset Overview")

st.write("Number of records:", len(df))
st.write("Number of variables:", len(df.columns))

st.dataframe(df)

# Risk level distribution
st.header("2. Risk Level Distribution")

risk_counts = df["RiskLevel"].value_counts()

st.bar_chart(risk_counts)

# Data quality
st.header("3. Data Quality")

st.write("Missing values:", df.isnull().sum().sum())
st.write("Duplicate rows:", df.duplicated().sum())

