import pickle
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
model = pickle.load(open('modelTest6.pkl', 'rb'))
email = "Hi Barbra. Attached is a summary of our meeting from Tuesday. Have a good weekend, John."
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
email_vec = vectorizer.transform([email])
prediction = model.predict(email_vec)
print(prediction)