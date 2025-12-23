import streamlit as st
import pickle

# Load model
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")

user_input = st.text_area("Enter News Headline or Content")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        vec_input = vectorizer.transform([user_input])
        prediction = model.predict(vec_input)[0]
        confidence = model.predict_proba(vec_input).max()

        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {confidence:.2f}")
