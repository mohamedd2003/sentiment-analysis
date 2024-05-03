
from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
import pandas as pd
import nltk
stopwords_nltk = nltk.corpus.stopwords
stop_words = stopwords_nltk.words('english')
model_pkl_file = "twitter_sentiment_analysis.pkl"  
with open(model_pkl_file, 'rb') as file:  
    model = pickle.load(file)
with open('vectorizer.pkl', 'rb') as file2:  
    loaded_vectorizer  = pickle.load(file2)


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    text = request.form.get('text', '')
    result = ''
    if text == '':
        pass
    else:
        
        result = model.predict(loaded_vectorizer.transform([text]))
        if result == 'Positive':
            result = '🥰'
        elif result == 'Negative':
            result = '😥'
        else:
            result = '🧐'
    return render_template('index.html', text=text, result=result)

def dpi(num):
    if num > 20:
        return f"{num}"
    else: return num

if __name__ == '__main__':
    app.run(debug=True)
