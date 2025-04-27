import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix
import io

model = load_model('cvd_model_with_pca.h5')

def preprocess_and_predict(uploaded_file):
    dataset = pd.read_csv(uploaded_file)
    
    st.write("Dataset Preview:")
    st.dataframe(dataset.head())
    
    if dataset.shape[1] < 2:
        st.error("Dataset must have more than 1 column.")
        return
    
    X = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1].values
    
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    predictions = model.predict(X_scaled)
    predictions = (predictions > 0.5).astype(int)
    
    dataset['Predicted_CVD'] = predictions
    
    st.write("Updated Dataset with Predictions:")
    st.dataframe(dataset)

    accuracy = accuracy_score(y, predictions)
    cm = confusion_matrix(y, predictions)

    st.write(f"Accuracy of the model on this dataset: {accuracy * 100:.2f}%")
    
    st.subheader("Confusion Matrix:")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No CVD', 'CVD'], yticklabels=['No CVD', 'CVD'], ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)
    
    st.subheader("Prediction Distribution:")
    prediction_df = pd.DataFrame(predictions, columns=['Predicted_CVD'])
    fig = px.histogram(prediction_df, x='Predicted_CVD', title="Distribution of Predictions")
    st.plotly_chart(fig)
    
    st.subheader("Download Predicted CSV:")
    csv = dataset.to_csv(index=False)
    st.download_button(
        label="Download Predicted Data",
        data=csv,
        file_name='predicted_cvd_data.csv',
        mime='text/csv'
    )

st.title('CVD Prediction App')
st.subheader('Upload your dataset and predict CVD')

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    preprocess_and_predict(uploaded_file)
