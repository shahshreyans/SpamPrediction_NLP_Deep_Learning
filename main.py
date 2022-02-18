import string

import numpy as np
import pandas as pd
import sklearn
from nltk import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.translate import metrics
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

import nltk
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def main():
    print("main Function")


if __name__ == "__main__":
    main()
dataset = pd.read_csv(r'C:\Users\Shreyans\PycharmProjects\spamdetection\SMS_train.csv', encoding='unicode_escape')
print(dataset)
test_dataset = pd.read_csv(r'C:\Users\Shreyans\PycharmProjects\spamdetection\SMS_test.csv', encoding='unicode_escape')


x_new = dataset.Message_body
y_new = dataset.Label
x_new_test = test_dataset.Message_body
y_new_test = test_dataset.Label

#data preprocessing
# new_dataset =dataset
# stemer = SnowballStemmer(language='english')
# lem = WordNetLemmatizer()
# for message in dataset.Message_body:
#     new_message = message.lower()
#     new_message = message.translate(str.maketrans('', '', string.punctuation))
#     new_message_list = [word for word in new_message.split() if word not in  stopwords.words('english')]
#     new_message_list = [i for i in new_message.split()]
#     new_message_list = [lem.lemmatize(word) for word in new_message_list]
#     new_message_list = [stemer.stem(word) for word in new_message_list]
#     new_message = str(" ".join(new_message_list))
#     new_dataset.Message_body = dataset.Message_body.replace(to_replace=message, value=new_message)
# print(new_dataset)

#Multinominal Naive Bias
X_train, X_test, y_train, y_test = train_test_split(x_new, y_new,test_size=1,random_state=1,shuffle=True)
count_vect = CountVectorizer(stop_words='english',analyzer='word')
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)
y_pred = clf.predict(count_vect.transform(x_new_test))
print('MultiNominal Bias (Accuracy Score):',accuracy_score(y_new_test, y_pred))
#print(clf.predict(count_vect.transform(["U get 3000 reward point"])))
print(sklearn.metrics.classification_report(y_new_test,y_pred,target_names=dataset['Label'].unique()))

#Liner Svc
model = LinearSVC()
X_train, X_test, y_train, y_test, = train_test_split(x_new, y_new,test_size=1 ,random_state=0,shuffle=True)
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(count_vect.transform(x_new_test))
print('Linear SVC (Accuracy Score):',accuracy_score( y_new_test,y_pred))
#print(model.predict(count_vect.transform(["You will get 3000 reward point"])))
print(sklearn.metrics.classification_report(y_new_test,y_pred,target_names=dataset['Label'].unique()))

#KNN
model3 = KNeighborsClassifier(n_neighbors=1, metric = "euclidean")
X_train, X_test, y_train, y_test, = train_test_split(x_new, y_new,test_size=1,random_state=0,shuffle=True)
model3.fit(X_train_tfidf,y_train)
y_pred = model3.predict(count_vect.transform(x_new_test))
print('KNN(Accuracy Score):',accuracy_score(y_new_test,y_pred))
#print(model3.predict(count_vect.transform(["You will get 3000 reward point"])))
print(sklearn.metrics.classification_report(y_new_test,y_pred,target_names=dataset['Label'].unique()))

#SVC
model4 = SVC(kernel ='linear')
X_train, X_test, y_train, y_test, = train_test_split(x_new, y_new,test_size=1 ,random_state=0,shuffle=True)
model4.fit(X_train_tfidf,y_train)
y_pred = model4.predict(count_vect.transform(x_new_test))
print('SVC:',accuracy_score(y_new_test,y_pred))
#print(model4.predict(count_vect.transform(["You will get 3000 reward point"])))
print(sklearn.metrics.classification_report(y_new_test,y_pred,target_names=dataset['Label'].unique()))

#RFC
model5 = RandomForestClassifier(n_estimators=30, criterion = "entropy", random_state=0)
X_train, X_test, y_train, y_test, = train_test_split(x_new, y_new,test_size=1 ,random_state=0,shuffle=True)
model5.fit(X_train_tfidf,y_train)
y_pred = model5.predict(count_vect.transform(x_new_test))
print('RFC(Accuracy Score):',accuracy_score(y_new_test,y_pred))
#print(model5.predict(count_vect.transform(["You will get 3000 reward point"])))
print(sklearn.metrics.classification_report(y_new_test,y_pred,target_names=dataset['Label'].unique()))

#DTC
model6 = DecisionTreeClassifier(criterion = "entropy",max_features='auto')
X_train, X_test, y_train, y_test, = train_test_split(x_new, y_new,test_size=1 ,random_state=0,shuffle=True)
model6.fit(X_train_tfidf,y_train)
y_pred = model6.predict(count_vect.transform(x_new_test))
print('DTC(Accuracy Score):',accuracy_score(y_new_test,y_pred) )
#print(model6.predict(count_vect.transform(["You will get 3000 reward point"])))
print(sklearn.metrics.classification_report(y_new_test,y_pred,target_names=dataset['Label'].unique()))


