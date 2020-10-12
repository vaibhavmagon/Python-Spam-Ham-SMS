import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

messages = pd.read_csv('data/spam-ham.csv', delimiter=',\t', names=['label', 'sms'], engine='python')

messages_real = pd.read_csv('data/spam-real.csv', names=['sms'])

test_data = messages_real.iloc[:, 0]

messages['label'] = messages.label.map({'ham': 0, 'spam': 1})  # Making spam/ham as 0/1 instead

y_train = pd.Series(messages['label'])


# Cleaning the data
def dataCleanFunc(data):
    corpus = []
    for i in range(0, len(data)):
        smsDocument = re.sub('[^a-zA-Z]', ' ', data[i])
        smsDocument = smsDocument.lower()
        smsDocument = smsDocument.split()
        ps = PorterStemmer()
        smsDocument = [ps.stem(word) for word in smsDocument if not word in set(stopwords.words('english'))]
        smsDocument = ' '.join(smsDocument)
        corpus.append(smsDocument)
    return corpus


training_data = dataCleanFunc(messages['sms'])

training_data = pd.Series(training_data)

count_vector = CountVectorizer(max_features=1500)  # Making document term matrix / frequency matrix

training_data = count_vector.fit_transform(training_data)

testing_data = count_vector.transform(test_data)

# Multinomail distribution (naive bayes classification) works better in case of counts.
model = MultinomialNB()

model.fit(training_data, y_train)

y_score = model.predict(testing_data)
y_score = pd.Series(y_score, name='y_score')

final = y_score.to_frame().join(test_data.to_frame())

final.to_csv('data/test-spam.csv', index=False)
