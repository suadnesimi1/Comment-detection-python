import os
import pandas as pd
import regex as re
import fnmatch
import nltk.corpus
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from textblob import TextBlob

path = "C:/Users/AdwNess/Desktop/DataMining/opinion_spam/negative"

label = []

config_files = [os.path.join(subdir, f)

                for subdir, dirs, files in os.walk(path)
                for f in fnmatch.filter(files, "*.txt")]
print(config_files[0])

for f in config_files:
    des = re.search("(deceptive|truthful)", f)
    label.append(des.group())

labels = pd.DataFrame(label, columns=['Labels'])
print(len(config_files))

review = []
for subdir, dirs, files in os.walk(path):
    for file in files:
        if fnmatch.filter(files, "*.txt"):
            f = open(os.path.join(subdir, file), 'r')
            f = f.read()
            review.append(f)

reviews = pd.DataFrame(review, columns=["Review"])

customer_review = pd.merge(reviews, labels, right_index=True, left_index=True)
customer_review.Review = customer_review.Review.map(lambda x: x.lower())

stop_words = stopwords.words("English")
customer_review['review_without_stopwords'] = customer_review['Review'].apply(
    lambda x: ' '.join([word for word in x.split()

                        if word not in stop_words]))

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def postagg(review):
    return TextBlob(review).tags


poss = customer_review.review_without_stopwords.apply(postagg)
poss = pd.DataFrame(poss)

poss["part_of_speech"] = poss['review_without_stopwords'].map(lambda x: " ".join(["/".join(x) for x in x]))
customer_review = pd.merge(customer_review, poss, right_index=True, left_index=True)
train_review, test_review, train_label, test_label = train_test_split(customer_review.part_of_speech,
                                                                      customer_review.Labels,
                                                                      test_size=0.25,
                                                                      random_state=10)
tf_vect = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True, sublinear_tf=False)
X_train = tf_vect.fit_transform(train_review)
X_test = tf_vect.transform(test_review)


def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_


svc_param_selection(X_train, train_label, 6)
clf = svm.SVC(C=1, gamma=0.001, kernel="linear")
clf.fit(X_train, train_label)
pred = clf.predict(X_test)
print(accuracy_score(pred, test_label))


print(classification_report(pred, test_label))
