import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ✅ Ensure NLTK data is downloaded (Fixes "stopwords not found" error)
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)

nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

# Initialize stemmer
ps = PorterStemmer()

# ✅ Function to clean and transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]  # Remove non-alphanumeric characters

    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]  # Remove stopwords & punctuation

    y = [ps.stem(i) for i in y]  # Apply stemming

    return " ".join(y)

# ✅ Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ✅ Set page config
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# ✅ Custom header
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📩 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ✅ Input box
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

# ✅ Footer
st.markdown("""
---
<div style='text-align: center'>
    <small>Built By: Devendar Singh Rawat</small>
</div>
""", unsafe_allow_html=True)
