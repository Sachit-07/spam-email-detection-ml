import streamlit as st
import pickle
import nltk
from dotenv import load_dotenv
import os
import google.generativeai as genai
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))



from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("spam_model.pkl", "rb"))

def preprocess_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english"):
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    text = y[:]

    return " ".join(text)


gemini_model = genai.GenerativeModel("gemini-3.6-flash")


def explain_prediction(message, prediction):

    prompt = f"""
    The machine learning model has already classified this message as {prediction}.

    Do NOT classify the message again.

    Explain the machine learning prediction in simple language.

    Include:
    Respond in Markdown.

    Use these headings:

    ## 🤖 Why this message was classified

    ## 🚩 Suspicious phrases

    ## 🛡 Safety tips

    ## ✅ Recommended action

    Keep the explanation concise (150–250 words).

Do NOT classify the message again.
The machine learning prediction is already {prediction}.
    Message:
    {message}
    """
    response = gemini_model.generate_content(prompt)
    return response.text

st.title("Spam Email Detector")
st.write("Enter an email or SMS message below to check whether it is Spam or Ham.")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):

    transformed_sms = preprocess_text(input_sms)

    vector_input = vectorizer.transform([transformed_sms])

    result = model.predict(vector_input)

    if result[0] == 1:
        prediction = "Spam"
    else:
        prediction = "Ham"

    if prediction == "Spam":
     st.error(f"🚨 Prediction: {prediction}")
    else:
     st.success(f"✅ Prediction: {prediction}")

    with st.spinner("Generating AI explanation..."):
     explanation = explain_prediction(input_sms, prediction)

    explanation = explain_prediction(input_sms, prediction)
    

    st.subheader("🤖 AI Explanation")

    st.write(explanation)

