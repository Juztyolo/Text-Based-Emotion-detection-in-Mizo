# Mizo Emotion Classifier App 😐😠😂😔😨

A web-based application to classify **Mizo-language text** into five basic emotions: **joy**, **sadness**, **anger**, **fear**, and **neutral**. Built using **Streamlit**, it offers emotion prediction along with a probability distribution chart.

## 🚀 Features

- 📌 Predicts emotions from Mizo-language input text.
- 📊 Displays prediction confidence with a bar chart.
- 💻 User-friendly web interface with Streamlit.
- 🤖 Trained using a custom ML pipeline (`Mizo_emotion_classifier.pkl`).
- 🧠 Supports five emotion classes:
  - 😠 Anger
  - 😨 Fear
  - 😂 Joy
  - 😐 Neutral
  - 😔 Sadness

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas / NumPy
- Altair (for visualization)
- scikit-learn (via joblib model)

## 📁 Project Structure

├── Example App.py # Streamlit app
├── mizotext.csv # Dataset used for training/testing
├── Main.ipynb # Notebook with training workflow
├── Mizo_emotion_classifier.pkl # Pre-trained model (required at runtime)

## About
This project was created by Zorinsanga as part of a final year project. It aims to promote natural language processing (NLP) research in underrepresented languages like Mizo.
