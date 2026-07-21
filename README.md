# 📧 AI-Powered Spam Email Detector

An AI-powered Spam Email Detection system built using Machine Learning and Google's Gemini API.

## Features

- Spam/Ham prediction using Multinomial Naive Bayes
- Text preprocessing using NLTK
- TF-IDF feature extraction
- Interactive Streamlit web interface
- AI-generated explanation using Gemini
- Safety tips and suspicious phrase detection

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- NLTK
- Google Gemini API
- Pickle

## Project Structure

```
spam-email-detection-ml/
│
├── app.py
├── Spam_Email_Detector.ipynb
├── spam_model.pkl
├── vectorizer.pkl
├── spam.csv
├── requirements.txt
├── README.md
├── .env
└── .gitignore
```

## How to Run

1. Clone the repository

```
git clone <repository-url>
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Create a `.env` file

```
GEMINI_API_KEY=your_api_key_here
```

4. Start the application

```
streamlit run app.py
```

## Machine Learning Pipeline

```
Input Text
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Naive Bayes Classifier
     ↓
Spam/Ham Prediction
     ↓
Gemini AI Explanation
```

## Future Improvements

- Support multiple languages
- Email attachment analysis
- Confidence score visualization
- Phishing URL detection
- Email sender reputation analysis

## Author

Sachit S