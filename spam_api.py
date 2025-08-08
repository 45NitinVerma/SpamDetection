import nltk
import os
import pickle
import string
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.stem.porter import PorterStemmer

# Set up NLTK runtime path
nltk_path = os.path.join(os.getcwd(), 'nltk_data')
os.environ["NLTK_DATA"] = nltk_path
nltk.data.path.append(nltk_path)

# Force download required data (complete)
nltk.download('punkt', download_dir=nltk_path)
nltk.download('stopwords', download_dir=nltk_path)
from nltk.corpus import stopwords

from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
ps = PorterStemmer()

def transform_text(text):
    stopword_set = set(stopwords.words("english"))

    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopword_set and i not in string.punctuation:
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
