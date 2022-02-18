
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import string
import numpy as np

import pandas as pd
from nltk import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import nltk
from sklearn.linear_model import LogisticRegression


# main Function
def main():
    train_dataset, test_dataset = gettingDataset()

    x = 0
    while (x == 0):
        datapreprocess = input('Do you want to pre process the data?yes or no')
        datapreprocess.lower()
        if datapreprocess == 'yes':
            boool = True
        else:
            boool = False
        x_new, y_new = dataPreProcessing(train_dataset, boool)
        inputmessage = input('Enter Message you want to check')
        #classifier_multiNominalNaiveBias(x_new, y_new, inputmessage)
        classifier_Linersvc(x_new, y_new, inputmessage)
        con = input('Do you still want to check messages? Yes or No')
        con.lower()
        if con == 'yes':
            pass
        elif con == 'no':
            x = 1
        else:
            continue
    # inputmsg = input('Enter Message you want to upload to the databse')
    # inputlabel = input('Enter label : Spam or Non-Spam')
    # upload_data_to_dataset(inputmsg,inputlabel,train_dataset)
def gettingDataset():
    train_dataset = pd.read_csv(r'C:\Users\Shreyans\PycharmProjects\spamdetection\SMS_train.csv',
                                encoding='unicode_escape')
    test_dataset = pd.read_csv(r'C:\Users\Shreyans\PycharmProjects\spamdetection\SMS_test.csv',
                               encoding='unicode_escape')
    return train_dataset, test_dataset


def dataPreProcessing(dataset, bool):
    x_new = dataset.Message_body
    y_new = dataset.Label
    if bool == True:
        stemmer = SnowballStemmer(language='english')
        lemmatizer= WordNetLemmatizer()
        for message in x_new:
            new_message = message.lower()
            new_message = new_message.translate(str.maketrans('', '', string.punctuation))
            new_message_list = [word for word in new_message.split() if word not in stopwords.words('english')]
            #new_message_list=[word for word in new_message.split()]
            new_message_list = [lemmatizer.lemmatize(word) for word in new_message_list]
            new_message_list = [stemmer.stem(word) for word in new_message_list]
            new_message = str(" ".join(new_message_list))
            x_new = x_new.replace(to_replace=message, value=new_message)
    return x_new, y_new


def classifier_multiNominalNaiveBias(x_new, y_new, inputmessage):
    X_train, X_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.1, random_state=1)
    count_vector = CountVectorizer(stop_words='english', analyzer='word')
    X_train_counts = count_vector.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    #y_pred = clf.predict(count_vector.transform(X_test))
    #print(accuracy_score(y_test, y_pred))
    print(clf.predict(count_vector.transform([inputmessage])))


def classifier_Linersvc(x_new, y_new, inputmessage):
    model = LinearSVC()
    X_train, X_test, y_train, y_test, = train_test_split(x_new, y_new,test_size=0.1, random_state=0)
    count_vector = CountVectorizer(stop_words='english', analyzer='word')
    X_train_counts = count_vector.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    model.fit(X_train_tfidf, y_train)
    #y_pred = model.predict(count_vector.transform(X_test))
    #print(accuracy_score(y_test, y_pred))
    print(model.predict(count_vector.transform([inputmessage])))

# def upload_data_to_dataset(message,label,dataset):
#     data = {'Message_body':message,'Label':label}
#     dataset.append(data)
#     print(dataset)



if __name__ == "__main__":
    main()
