import streamlit as st
import pickle
import string
import nltk
import ssl
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -------------------------
# ✅ Fix SSL Issues (for NLTK download in cloud environments)
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# -------------------------
# ✅ Set up NLTK resources
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Download required resources if not already present
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

# -------------------------
# ✅ Initialize stemmer
ps = PorterStemmer()

# -------------------------
# ✅ Preprocessing Function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

# -------------------------
# ✅ Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# -------------------------
# ✅ Streamlit UI
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📩 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

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

        # 4. Show result
        if result == 1:
            st.error("🚨 This message is **Spam**!")
        else:
            st.success("✅ This message is **Not Spam**.")

# -------------------------
# ✅ Footer
st.markdown("""
---
<div style='text-align: center'>
    <small>Built By: Devendar Singh Rawat</small>
</div>
""", unsafe_allow_html=True)
