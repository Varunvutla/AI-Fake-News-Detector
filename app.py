import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("AI Fake News Detection System")

news = st.text_area("Enter News Article")

if st.button("Detect"):

    transformed_news = vectorizer.transform([news])

    prediction = model.predict(transformed_news)

    if prediction[0] == 1:
        st.error("Fake News Detected")
    else:
        st.success("Real News")