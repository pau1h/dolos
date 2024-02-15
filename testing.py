import pickle
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
model = pickle.load(open('model5.pkl', 'rb'))
email = 'This is a test of the model prediction'
vectorizer = TfidfVectorizer()
email_vec = vectorizer.transform([email])
print(model.predict(email_vec))