# !/usr/bin/env python

""" ml_rt.py: HelpDesk Request Tracker ML exploration"""

__author__ = "Mauricio Moldes"
__version__ = "0.1"
__maintainer__ = "Mauricio Moldes"
__email__ = "mauricio.moldes@crg.eu"
__status__ = "Developement"

import logging
import pandas as pd
import numpy as np  # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk import word_tokenize
from collections import Counter
import base64
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction import _stop_words
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import string
import re
import spacy

spacy.load('en_core_web_sm')
from spacy.lang.en import English

parser = English()
from sklearn import metrics

logger = logging.getLogger('rt_ml')

""" READ DATA """


def read_data(path):
    data = pd.read_csv(path)
    return data


""" SPLIT DATA INTO TRAINING SET """


def split_data_training(df):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.33, random_state=42)
    print('message sample:', train['message'].iloc[0])
    print('tag of this message:', train['tag'].iloc[0])
    print('Training Data Shape:', train.shape)
    print('Testing Data Shape:', test.shape)
    return train, test


def find_top_words(train):
    import spacy
    import string
    from nltk.corpus import stopwords
    nlp = spacy.load('en_core_web_sm')
    punctuations = string.punctuation
    stop_words = set(stopwords.words("english"))

    def cleanup_text(docs, logging=False):
        texts = []
        counter = 1
        for doc in docs:
            if counter % 1000 == 0 and logging:
                print("Processed %d out of %d documents." % (counter, len(docs)))
            counter += 1
            doc = nlp(doc, disable=['parser', 'ner'])
            tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
            tokens = [tok for tok in tokens if tok not in stop_words and tok not in punctuations]
            tokens = ' '.join(tokens)
            texts.append(tokens)
        return pd.Series(texts)

    # Common words for Data Deposition
    INFO_text = [text for text in train[train['tag'] == 'Data deposition intention']['message']]
    INFO_clean = cleanup_text(INFO_text)
    INFO_clean = ' '.join(INFO_clean).split()
    INFO_counts = Counter(INFO_clean)
    INFO_common_words = [word[0] for word in INFO_counts.most_common(20)]
    INFO_common_counts = [word[1] for word in INFO_counts.most_common(20)]
    fig = plt.figure(figsize=(18, 6))
    sns.barplot(x=INFO_common_words, y=INFO_common_counts)
    plt.title('Most Common Words used in tickets for data deposition intention')
    # plt.show()
    fig.savefig('Data_deposition_intention.pdf')

    # Common words for Missing Study Dataset
    IS_text = [text for text in train[train['tag'] == 'Missing study/dataset on live']['message']]
    IS_clean = cleanup_text(IS_text)
    IS_clean = ' '.join(IS_clean).split()
    IS_counts = Counter(IS_clean)
    IS_common_words = [word[0] for word in IS_counts.most_common(20)]
    IS_common_counts = [word[1] for word in IS_counts.most_common(20)]
    fig = plt.figure(figsize=(18, 6))
    sns.barplot(x=IS_common_words, y=IS_common_counts)
    plt.title('Most Common Words used in tickets for Missing study/dataset on live')
    # plt.show()
    fig.savefig('Missing_study_dataset_on_live.pdf')

    # Common words for Missing Study Dataset
    IS_text = [text for text in train[train['tag'] == 'Study/Dataset Release']['message']]
    IS_clean = cleanup_text(IS_text)
    IS_clean = ' '.join(IS_clean).split()
    IS_counts = Counter(IS_clean)
    IS_common_words = [word[0] for word in IS_counts.most_common(20)]
    IS_common_counts = [word[1] for word in IS_counts.most_common(20)]
    fig = plt.figure(figsize=(18, 6))
    sns.barplot(x=IS_common_words, y=IS_common_counts)
    plt.title('Most Common Words used in tickets for study_dataset_release')
    # plt.show()
    fig.savefig('study_dataset_release.pdf')

    # Common words for Missing Study Dataset
    IS_text = [text for text in train[train['tag'] == 'DAC request to create accounts']['message']]
    IS_clean = cleanup_text(IS_text)
    IS_clean = ' '.join(IS_clean).split()
    IS_counts = Counter(IS_clean)
    IS_common_words = [word[0] for word in IS_counts.most_common(20)]
    IS_common_counts = [word[1] for word in IS_counts.most_common(20)]
    fig = plt.figure(figsize=(18, 6))
    sns.barplot(x=IS_common_words, y=IS_common_counts)
    plt.title('Most Common Words used in tickets for DAC request to create accounts')
    # plt.show()
    fig.savefig('dac_request_create_accounts.pdf')

    # Common words for Data Deposition
    INFO_text = [text for text in train[train['tag'] == 'Data deposition']['message']]
    INFO_clean = cleanup_text(INFO_text)
    INFO_clean = ' '.join(INFO_clean).split()
    INFO_counts = Counter(INFO_clean)
    INFO_common_words = [word[0] for word in INFO_counts.most_common(20)]
    INFO_common_counts = [word[1] for word in INFO_counts.most_common(20)]
    fig = plt.figure(figsize=(18, 6))
    sns.barplot(x=INFO_common_words, y=INFO_common_counts)
    plt.title('Most Common Words used in tickets for data deposition')
    # plt.show()
    fig.savefig('Data_deposition.pdf')


class CleanTextTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self


def get_params(self, deep=True):
    return {}


def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text


def tokenizeText(sample):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    return tokens


def printNMostInformative(vectorizer, clf, N):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("Class 1 best: ")
    for feat in topClass1:
        print(feat)
    print("Class 2 best: ")
    for feat in topClass2:
        print(feat)


if __name__ == '__main__':
    data = read_data('../data/test.csv')
    NO_OF_ENTRIES = len(data)
    train, test = split_data_training(data)
    find_top_words(train)

    import spacy
    import string
    from nltk.corpus import stopwords

    nlp = spacy.load('en_core_web_sm')
    punctuations = string.punctuation
    stop_words = set(stopwords.words("english"))

    STOPLIST = set(stopwords.words('english'))
    SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]

    vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1, 1))
    clf = LinearSVC()

    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
    # data
    train1 = train['message'].tolist()
    labelsTrain1 = train['tag'].tolist()
    test1 = test['message'].tolist()
    labelsTest1 = test['tag'].tolist()
    # train
    pipe.fit(train1, labelsTrain1)
    # test
    preds = pipe.predict(test1)
    print("accuracy:", accuracy_score(labelsTest1, preds))
    print("Top 10 features used to predict: ")

    printNMostInformative(vectorizer, clf, 10)
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer)])
    transform = pipe.fit_transform(train1, labelsTrain1)
    vocab = vectorizer.get_feature_names()
    for i in range(len(train1)):
        s = ""
        indexIntoVocab = transform.indices[transform.indptr[i]:transform.indptr[i + 1]]
        numOccurences = transform.data[transform.indptr[i]:transform.indptr[i + 1]]
        for idx, num in zip(indexIntoVocab, numOccurences):
            s += str((vocab[idx], num))

    print(metrics.classification_report(labelsTest1, preds))
