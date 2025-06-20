import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
import streamlit as st


# Load the saved model
kmeans_model = joblib.load('kmeans_model.pkl')
# Load the dataset into a DataFrame
df = pd.read_csv('Mall_Customers.csv')
# Extract the features for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
#X_array = X.values
# Set up the Streamlit app
st.set_page_config(page_title="Customer Clustering", layout="wide")
st.title("Customer Clustering App")
st.write("Enter the annual income and spending score to predict the customer cluster.")
st.image("cluster_plot.png")
#inputs
income = st.number_input("Annual Income (k$)", min_value=0, max_value=400, value=50)
spending_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)

#button for the prediction
if st.button("Predict Cluster"):
    input_data = np.array([[income, spending_score]])
    cluster = kmeans_model.predict(input_data)
    st.success(f"The predicted cluster is: {cluster}")



