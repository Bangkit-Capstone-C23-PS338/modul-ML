import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from statistics import mode
from nltk.tokenize import word_tokenize

global model
with open('model/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

global df_cleaned
with open('model/df_cleaned.pkl', 'rb') as f:
    df_cleaned = pickle.load(f)

cv = CountVectorizer()
cv.fit_transform(df_cleaned)

# variable "data" is the independent variable. Feel free to change it
data = ["bagus banget", 'jelek banget']

vect = cv.transform(data).toarray()

prediction = model.predict_proba(vect)
results = prediction[:, 1]