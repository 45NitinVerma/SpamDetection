import nltk
import os

# Set up NLTK path for Render (ephemeral but usable during runtime)
nltk_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_path)

if not os.path.exists(nltk_path):
    os.makedirs(nltk_path, exist_ok=True)

# Safe download with check
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_path)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import string
from nltk.corpus import stopwords
stopwords.words('english')
from nltk.stem.porter import PorterStemmer


app = Flask(__name__)
CORS(app)

# Load model and vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

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

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        msg = data["message"]
        transformed = transform_text(msg)
        vector_input = tfidf.transform([transformed])
        result = model.predict(vector_input)[0]
        return jsonify({"prediction": "Spam" if result == 1 else "Not Spam"})
    except Exception as e:
        print("🔥 Error in /predict:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
