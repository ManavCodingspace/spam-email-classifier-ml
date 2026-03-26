import pandas as pd
import numpy as np 
import matplotlib.pylab as plt
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

df = pd.read_csv("spam.csv", encoding='latin-1')
df.columns = df.columns.str.strip()

print(df.head())
print(df.info())

columns = ['v1','v2','Unnamed: 2','Unnamed: 3','Unnamed: 4']
df = df.drop(columns =['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
df['v2'] = df['v2'].str.lower()
translator = str.maketrans("","",string.punctuation)
df["v2"] = df["v2"].str.translate(translator)

print(df.info())
print(df.head())


y = df['v1']

le = LabelEncoder()
y = le.fit_transform(y)

vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
x = vectorizer.fit_transform(df["v2"])

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=42)

model = MultinomialNB(class_prior=None)
model = model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

