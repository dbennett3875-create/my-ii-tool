import streamlit as st
import pandas as pd

st.set_page_config(page_title="Test I&I Tool", layout="wide")

st.title("🔧 I&I Analysis Tool - Test Version")
st.write("If you can see this, Streamlit is working!")

# Simple file uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} rows!")
    st.dataframe(df.head())
else:
    st.info("Upload a file to get started")
