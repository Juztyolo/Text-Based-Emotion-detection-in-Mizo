# Core Pkgs
import streamlit as st
import altair as alt

# EDA Pkgs
import pandas as pd
import numpy as np

# Utils
import joblib

pipe_lr = joblib.load(open("Mizo_emotion_classifier.pkl", "rb"))


# Fxn
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


emotions_emoji_dict = {"anger": "😠", "fear": "😨😱", "joy": "😂", "neutral": "😐",
                        "sadness": "😔"}


# Main Application
def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Emotion In Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            # Apply Fxn Here
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)


    elif choice == "About":
        st.subheader("About")
        st.write('This app was made so that users can test out the '
                 'emotion classifier for mizo language made by Zorinsanga')
        st.write('This app can detect five emotion,they are '
                 'joy,sadness,anger,fear and neutral as well as show a probability chart of the text')



if __name__ == '__main__':
    main()
