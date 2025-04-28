import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import speech_recognition as sr

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# Mobile-friendly styling
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    @media only screen and (max-width: 600px) {
        .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Preprocessing without punkt
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def analyze_sentiment(text):
    processed_text = preprocess_text(text)
    score = sia.polarity_scores(processed_text)["compound"]
    label = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
    return processed_text, score, label

# App title
st.title("📱 Sentiment Analysis App (Text | CSV | Speech)")

# Upload section
st.header("📂 Upload CSV or TXT")
uploaded_file = st.file_uploader("Upload your file", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            content = uploaded_file.getvalue().decode("utf-8")
            df = pd.DataFrame({"Text": content.splitlines()})

        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head())

        # Detect text column
        text_column = None
        for col in df.columns:
            if df[col].dtype == "object":
                text_column = col
                break

        if text_column:
            df[["Processed_Text", "Sentiment_Score", "Sentiment_Label"]] = df[text_column].apply(
                lambda x: pd.Series(analyze_sentiment(str(x)))
            )
            st.subheader("✅ Sentiment Analysis Results")
            st.dataframe(df[[text_column, "Processed_Text", "Sentiment_Label"]])

            st.subheader("📊 Sentiment Distribution")
            fig, ax = plt.subplots()
            df["Sentiment_Label"].value_counts().plot(kind="bar", ax=ax, color=["green", "red", "gray"])
            st.pyplot(fig)
        else:
            st.warning("No text column found.")

    except Exception as e:
        st.error(f"Error reading file: {e}")

# Text input
st.header("📝 Real-time Text Sentiment")
user_text = st.text_area("Enter a comment or message:")

if user_text:
    processed, score, label = analyze_sentiment(user_text)
    st.success(f"**Sentiment:** {label}")
    st.write(f"**Processed:** {processed}")
    st.write(f"**Score:** {score}")

# Speech input
st.header("🎤 Real-time Speech Sentiment")
if st.button("🎙️ Start Recording"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak now...")
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            st.write(f"**You said:** {text}")
            processed, score, label = analyze_sentiment(text)
            st.success(f"**Sentiment:** {label}")
            st.write(f"**Processed:** {processed}")
            st.write(f"**Score:** {score}")
        except sr.UnknownValueError:
            st.error("Speech not understood.")
        except sr.RequestError as e:
            st.error(f"Speech service error: {e}")