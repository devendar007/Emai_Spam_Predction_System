import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer

# Hardcoded English stopwords list (safe for deployment)
stop_words = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','any','both','each','few','more',
    'most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','s','t','can','will','just','don','should','now'
])

# Initialize stemmer
ps = PorterStemmer()

# Function to clean and transform text
def transform_text(text):
    text = text.lower()
    text = text.split()  # replaces nltk.word_tokenize

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stop_words]
    y = [ps.stem(i) for i in y]

    return " ".join(y)


# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit config
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# App title
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📩 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Input
input_sms = st.text_area("✉️ Enter the message you want to classify", height=150)

if st.button('🚀 Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Preprocess, vectorize, predict
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

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
