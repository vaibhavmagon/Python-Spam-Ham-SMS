import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
import re
import nltk
# nltk.download('stopwords') #used once to download stopwords.
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# https://medium.com/coinmonks/spam-detector-using-naive-bayes-c22cc740e257

messages = pd.read_csv('data/spam-ham.csv', delimiter=',\t', names=['label', 'sms'])

messages['label'] = messages.label.map({'ham':0, 'spam':1}) #Making spam/ham as 0/1 instead

# Cleaning the data
corpus = []
for i in range(0,5574):
    smsDocument = re.sub('[^a-zA-Z]',' ',messages['sms'][i])
    smsDocument = smsDocument.lower()
    smsDocument = smsDocument.split()
    ps = PorterStemmer()
    smsDocument = [ps.stem(word) for word in smsDocument if not word in set(stopwords.words('english'))]
    smsDocument = ' '.join(smsDocument)
    corpus.append(smsDocument)

smsDocuments = pd.Series(corpus)

################### Summary #####################

count_vector = CountVectorizer()

'''
CountVectorizer will convert all the text into lowercase, will remove all the punctuations and all the stopwords and make
a frequency matrix similar to document term matrix.
'''

count_vector.fit(smsDocuments) #Bag of words algorithm.

'''
The basic idea of BoW is to take a piece of text and count the frequency of the words in that text. 
It is important to note that the BoW concept treats each word individually and the order in which 
the words occur does not matter.
'''

count_vector.get_feature_names()

doc_array = count_vector.transform(smsDocuments).toarray()

'''
Now we have a clean representation of the documents in terms of the frequency distribution of the words in them.
'''

freq_matrix = pd.DataFrame(doc_array, columns = count_vector.get_feature_names())


################## Real Detector ################


count_vector = CountVectorizer(max_features = 1500) #Making document term matrix / frequency matrix

smsDocuments = count_vector.fit_transform(smsDocuments)

training_data, testing_data, y_train, y_test = train_test_split(smsDocuments, messages['label'], test_size=0.20, random_state=0)

model = MultinomialNB() #Multinomail distribution (naive bayes classification) works better in case of counts.

model.fit(training_data, y_train)

pred = model.predict(testing_data)

accs = accuracy_score(y_test, pred)

print("Accuracy Score: ", accs)

fpr, tpr, _ = metrics.roc_curve(y_test,  pred)

auc = metrics.roc_auc_score(y_test, pred) #Find AUC score for the data set based on LR

print("AUC Score: ", auc)

plt.plot(fpr,tpr,label="spam-ham.csv, auc="+str(auc))
plt.legend(loc=4)
plt.show()


