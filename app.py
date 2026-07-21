import streamlit as st
import pickle
import nltk



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

st.title("Spam Email Detector")
st.write("Enter an email or SMS message below to check whether it is Spam or Ham.")

input_sms = st.text_area("Enter the message")
if st.button("Predict"):
  transformed_sms = preprocess_text(input_sms)
  vector_input = vectorizer.transform([transformed_sms])
  result = model.predict(vector_input)
  if result[0] == 1:
    st.header("Spam")
  else:
    st.header("Ham")