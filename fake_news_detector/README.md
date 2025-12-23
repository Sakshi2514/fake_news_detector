# Fake News Detector using Machine Learning

## Project Overview
Fake News Detector is a machine learning-based application that classifies news headlines or content as **Fake** or **Real**. The project uses Natural Language Processing (NLP) techniques and a supervised learning model to analyze textual data and provide accurate predictions through an interactive user interface.

This project was developed as part of the **Internship Project Phase (2 Weeks)** and focuses on practical implementation, clarity, and interview readiness.

---

## Objective
To automatically detect fake news by analyzing text content using machine learning and NLP techniques.

---

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit

---

## Project Structure
fake_news_detector/
│
├── data/
│ └── news.csv
│
├── model/
│ ├── fake_news_model.pkl
│ └── tfidf_vectorizer.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md

---

## Dataset
- The dataset contains **100 news samples**
- Balanced classes: **50 REAL, 50 FAKE**
- Columns:
  - `text` – News headline or content
  - `label` – REAL or FAKE

---

## Steps Involved in Building the Project

1. Collected and prepared a labeled dataset
2. Cleaned and vectorized text using TF-IDF
3. Trained a Logistic Regression classification model
4. Saved the trained model and vectorizer
5. Built a Streamlit web interface for predictions
6. Displayed prediction results with confidence score

---

## How to Run the Project

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
