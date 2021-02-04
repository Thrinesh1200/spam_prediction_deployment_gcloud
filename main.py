from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import nltk
import string
from nltk.corpus import stopwords
def cleaning(msg):
    no_punc=[c for c in msg if c not in string.punctuation]
    no_punc=''.join(no_punc)
    return[word for word in no_punc.split() if word.lower() not in stopwords.words('english')]


# load the model from disk
filename = 'spam.pkl'
model = pickle.load(open(filename, 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        my_prediction = model.predict(data)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)