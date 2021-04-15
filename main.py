# -*- coding: utf-8 -*-
import nltk
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, auc
from unidecode import unidecode
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
from sklearn import svm
import seaborn as sn
import numpy as np

english_stopwords = nltk.corpus.stopwords.words("english")
if not len(english_stopwords):
    nltk.download("stopwords")

tfidf_vectorizer = TfidfVectorizer(stop_words=english_stopwords, strip_accents="ascii")
count_vectorizer = CountVectorizer( analyzer="word", stop_words=english_stopwords, strip_accents="ascii",)

def train_svm(data):
    X = extract_features(data["Item Description"])
    x = data["CTH"]
    clf = svm.LinearSVC()
    clf.fit(X, x)
    return clf


def extract_features(docs):
    features = count_vectorizer.fit_transform(docs)
    return features


def preprocess_text(text):
    text = text.lower()
    text = unidecode(text)
    text = remove_special_chars(text)
    text = remove_numbers(text)
    return text


def remove_special_chars(text):
    return re.sub(r"[^\w\s\']", "", text)


def remove_numbers(text):
    return re.sub(r"\d", "", text)


def build_classification_report(clf, test_data):
    y_true = test_data["CTH"]
    docs = map(preprocess_text, test_data["Item Description"])
    tfidf = count_vectorizer.transform(docs)
    y_pred = clf.predict(tfidf)

    show_heatmap(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return report


def show_heatmap(y_true, y_predict):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot()

    columns = np.unique(y_true)
    mtrx = confusion_matrix(y_true, y_predict)
    df = pd.DataFrame(mtrx, columns=columns, index=columns)
    df.index.name="Actual"
    df.columns.name = "Predicted"
    sn.heatmap(df, cmap='Blues', annot=True, ax=ax)
    ax.set_yticklabels(labels=columns, rotation=0)
    plt.show()

def cross_validation_report(clf, dataset):
    data = count_vectorizer.transform(dataset["Item Description"])
    target = dataset["CTH"]
    return model_selection.cross_val_score(clf, data, target)


train_data = pd.read_excel(os.path.join(os.getcwd(), "train.xlsx"), engine="openpyxl")
test_data = pd.read_excel(os.path.join(os.getcwd(), "test.xlsx"), engine="openpyxl")
if __name__ == "__main__":

    print("Training SVM...")
    clf = train_svm(train_data)
    print("SVM trained")

    print("Building reports...")
    print("Classification report:")
    print(build_classification_report(clf, test_data))
    print("----------")
    print("Cross-validation report:")
    print(cross_validation_report(clf, train_data))

