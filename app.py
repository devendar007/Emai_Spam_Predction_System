import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import ssl

# Fix SSL issues on some cloud servers
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# ✅ Download required NLTK resources (Uses default directory)
import os
import nltk

# Set a safe local directory for NLTK data
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Download only if not already downloaded
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)


# Initialize stemmer
ps = PorterStemmer()

# Function to clean and transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)  # ✅ Will work now!

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set page config
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# Custom header
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📩 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Input box
input_sms = st.text_area("✉️ Enter the message you want to classify", height=150)

if st.button('🚀 Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display result with emoji
        if result == 1:
            st.error("🚨 This message is **Spam**!")
        else:
            st.success("✅ This message is **Not Spam**.")

# Footer
st.markdown("""
---
<div style='text-align: center'>
    <small>Built By: Devendar Singh Rawat</small>
</div>
""", unsafe_allow_html=True)
