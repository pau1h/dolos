import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
from langdetect import detect
from sklearnex import patch_sklearn
patch_sklearn()



data = pd.read_csv('combined_data.csv') 
print(data.isna().sum())
#sanitizing data
def fetch_lang(text):
    try:
        return detect(text)
    except:
        return "unknown"

data["lang"] = data["text"].apply(fetch_lang) #makes a new column called lang that has the language used in each email of the training data
pos = data[data["lang"] != "en"].index 
only_en = data.drop(index=pos) #if the emails arent in english, drop the row
only_en["lang"].unique
#only_en["label"].value_counts().plot(kind="bar")

#training model
x_train,x_test, y_train, y_test = train_test_split(only_en['text'], only_en['label'], test_size=0.2)
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english') #should test accuracy without stop words
features_train = vectorizer.fit_transform(x_train)
features_test = vectorizer.transform(x_test)
model = svm.SVC()
model.fit(features_train, y_train)
print("Accuracy {}".format(model.score(features_test, y_test)))
#saving model
with open('modelTest6.pkl', 'wb') as f:
    pickle.dump(model,f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer,f)

