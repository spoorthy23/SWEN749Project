from database import fetch_bug_data
from database import fetch_feature_data
from database import fetch_rating_data
from database import fetch_user_experience_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.sparse import hstack
import numpy as np

import MySQLdb

split_size = 260
feature_split_size = 207

use_metadata = True

class classify:
    def __init__(self):
        db = MySQLdb.connect(host="127.0.0.1", user="root", passwd="student", db="sys")

        db.autocommit(True)
        db.begin()
        cur = db.cursor()
        self.db = db
        self.cur = cur

        stopwords_removal = 'stopwords_removal'
        lemmatized_comment = 'lemmatized_comment'
        stopwords_removal_lemmatization = 'stopwords_removal_lemmatization'
        original_comment = 'original_comment'

        MultinomialNB = 'MultinomialNB'
        Tree = 'Tree'
        LogisticRegression = 'LogisticRegression'

        nlp_type = original_comment
        classifier = LogisticRegression

        self.bug_train, self.bug_rating_dict, self.bug_senti_dict, self.bug_senti_pos_dict, self.bug_senti_neg_dict, self.bug_present_simple_dict, self.bug_past_simple_dict, self.bug_future_dict, self.bug_present_con_dict, self.bug_cls, self.not_bug_cls = fetch_bug_data(
            self, nlp_type)
        self.feature_train, self.feature_rating_dict, self.feature_senti_dict, self.feature_senti_pos_dict, self.feature_senti_neg_dict, self.feature_present_simple_dict, self.feature_past_simple_dict, self.feature_future_dict, self.feature_present_con_dict, self.feature_cls, self.not_feature_cls = fetch_feature_data(
            self, nlp_type)
        self.rating_train, self.rating_rating_dict, self.rating_senti_dict, self.rating_senti_pos_dict, self.rating_senti_neg_dict, self.rating_present_simple_dict, self.rating_past_simple_dict, self.rating_future_dict, self.rating_present_con_dict, self.rating_cls, self.not_rating_cls = fetch_rating_data(
            self, nlp_type)
        self.user_experience_train, self.user_experience_rating_dict, self.user_experience_senti_dict, self.user_experience_senti_pos_dict, self.user_experience_senti_neg_dict, self.user_experience_present_simple_dict, self.user_experience_past_simple_dict, self.user_experience_future_dict, self.user_experience_present_con_dict, self.user_experience_cls, self.not_user_experience_cls = fetch_user_experience_data(
            self, nlp_type)

        self.db.commit()
        self.db.close()

        bugs_data, bugs_target, not_bugs_data, not_bugs_target = split_data(self.bug_train, self.bug_cls, self.not_bug_cls)
        user_experience_data, user_experience_target, not_user_experience_data, not_user_experience_target = split_data(self.user_experience_train, self.user_experience_cls, self.not_user_experience_cls)
        rating_data, rating_target, not_rating_data, not_rating_target = split_data(self.rating_train, self.rating_cls, self.not_rating_cls)
        feature_data, feature_target, not_feature_data, not_feature_target = split_data(self.feature_train, self.feature_cls, self.not_feature_cls)

        bug_precision, bug_recall, bug_f1_score = bug_classify(bugs_data, bugs_target, not_bugs_data, not_bugs_target, self.bug_rating_dict, self.bug_senti_dict, self.bug_senti_pos_dict, self.bug_senti_neg_dict, self.bug_present_simple_dict, self.bug_past_simple_dict, self.bug_future_dict, self.bug_present_con_dict, classifier)
        print("Bug: ", bug_precision, bug_recall, bug_f1_score)
        user_experience_precision, user_experience_recall, user_experience_f1_score = user_experience_classify(user_experience_data, user_experience_target, not_user_experience_data, not_user_experience_target, self.user_experience_rating_dict, self.user_experience_senti_dict, self.user_experience_senti_pos_dict, self.user_experience_senti_neg_dict, self.user_experience_present_simple_dict, self.user_experience_past_simple_dict, self.user_experience_future_dict, self.user_experience_present_con_dict, classifier)
        print("user Experience: ", user_experience_precision, user_experience_recall, user_experience_f1_score)
        rating_precision, rating_recall, rating_f1_score = rating_classify(rating_data, rating_target, not_rating_data, not_rating_target, self.rating_rating_dict, self.rating_senti_dict, self.rating_senti_pos_dict, self.rating_senti_neg_dict, self.rating_present_simple_dict, self.rating_past_simple_dict, self.rating_future_dict, self.rating_present_con_dict, classifier)
        print("Rating: ", rating_precision, rating_recall, rating_f1_score)
        feature_precision, feature_recall, feature_f1_score = feature_classify(feature_data, feature_target, not_feature_data, not_feature_target, self.feature_rating_dict, self.feature_senti_dict, self.feature_senti_pos_dict, self.feature_senti_neg_dict, self.feature_present_simple_dict, self.feature_past_simple_dict, self.feature_future_dict, self.feature_present_con_dict, classifier)
        print("Feature Request: ", feature_precision, feature_recall, feature_f1_score)

        rating_dict = {**self.bug_rating_dict, **self.feature_rating_dict, **self.rating_rating_dict, **self.user_experience_rating_dict}
        senti_dict = {**self.bug_senti_dict, **self.feature_senti_dict, **self.rating_senti_dict, **self.user_experience_senti_dict}
        senti_pos_dict = {**self.bug_senti_pos_dict, **self.feature_senti_pos_dict, **self.rating_senti_pos_dict, **self.user_experience_senti_pos_dict}
        senti_neg_dict = {**self.bug_senti_neg_dict, **self.feature_senti_neg_dict, **self.rating_senti_neg_dict, **self.user_experience_senti_neg_dict}
        present_simple_dict = {**self.bug_present_simple_dict, **self.feature_present_simple_dict, **self.rating_present_simple_dict, **self.user_experience_present_simple_dict}
        past_simple_dict = {**self.bug_past_simple_dict, **self.feature_past_simple_dict, **self.rating_past_simple_dict, **self.user_experience_past_simple_dict}
        future_dict = {**self.bug_future_dict, **self.feature_future_dict, **self.rating_future_dict, **self.user_experience_future_dict}
        present_con_dict = {**self.bug_present_con_dict, **self.feature_present_con_dict, **self.rating_present_con_dict, **self.user_experience_present_con_dict}

        multiclass_precision, multiclass_recall, multiclass_f1_score = multiclass_classify(bugs_data, bugs_target, user_experience_data, user_experience_target, rating_data, rating_target, feature_data, feature_target, rating_dict, senti_dict, senti_pos_dict, senti_neg_dict, present_simple_dict, past_simple_dict, future_dict, present_con_dict, classifier)
        print("Multiclass Classification: ", multiclass_precision, multiclass_recall, multiclass_f1_score)

def bug_classify(bugs_data, bugs_target, not_bugs_data, not_bugs_target, bug_rating_dict, bug_senti_dict, bug_senti_pos_dict, bug_senti_neg_dict, bug_present_simple_dict, bug_past_simple_dict, bug_future_dict, bug_present_con_dict, classifier):
    bugs_data_train, bugs_target_train, bugs_data_test, bugs_target_test = prepare_data_for_binary_classification(bugs_data, bugs_target, not_bugs_data, not_bugs_target, is_feature=False)
    bugs_tfidf_train_data, bugs_tfidf_test_data = vectorize_data(bugs_data_train, bugs_data_test, bug_rating_dict, bug_senti_dict, bug_senti_pos_dict, bug_senti_neg_dict, bug_present_simple_dict, bug_past_simple_dict, bug_future_dict, bug_present_con_dict)
    predicted_bugs_target_test = classify_app_reviews(classifier, bugs_tfidf_train_data, bugs_target_train, bugs_tfidf_test_data)
    precision, recall, f1_score = calculate_classifier_performance_metrics(bugs_target_test, predicted_bugs_target_test, pos_label=1)
    return precision, recall, f1_score


def user_experience_classify(user_experience_data, user_experience_target, not_user_experience_data, not_user_experience_target, user_experience_rating_dict, user_experience_senti_dict, user_experience_senti_pos_dict, user_experience_senti_neg_dict, user_experience_present_simple_dict, user_experience_past_simple_dict, user_experience_future_dict, user_experience_present_con_dict, classifier):
    user_experience_data_train, user_experience_target_train, user_experience_data_test, user_experience_target_test = prepare_data_for_binary_classification(user_experience_data, user_experience_target, not_user_experience_data, not_user_experience_target, is_feature=False)
    user_experience_tfidf_train_data, user_experience_tfidf_test_data = vectorize_data(user_experience_data_train, user_experience_data_test, user_experience_rating_dict, user_experience_senti_dict, user_experience_senti_pos_dict, user_experience_senti_neg_dict, user_experience_present_simple_dict, user_experience_past_simple_dict, user_experience_future_dict, user_experience_present_con_dict)
    predicted_user_experience_target_test = classify_app_reviews(classifier, user_experience_tfidf_train_data, user_experience_target_train, user_experience_tfidf_test_data)
    precision, recall, f1_score = calculate_classifier_performance_metrics(user_experience_target_test, predicted_user_experience_target_test, pos_label=3)
    return precision, recall, f1_score


def rating_classify(rating_data, rating_target, not_rating_data, not_rating_target, rating_rating_dict, rating_senti_dict, rating_senti_pos_dict, rating_senti_neg_dict, rating_present_simple_dict, rating_past_simple_dict, rating_future_dict, rating_present_con_dict, classifier):
    rating_data_train, rating_target_train, rating_data_test, rating_target_test = prepare_data_for_binary_classification(rating_data, rating_target, not_rating_data, not_rating_target, is_feature=False)
    rating_tfidf_train_data, rating_tfidf_test_data = vectorize_data(rating_data_train, rating_data_test, rating_rating_dict, rating_senti_dict, rating_senti_pos_dict, rating_senti_neg_dict, rating_present_simple_dict, rating_past_simple_dict, rating_future_dict, rating_present_con_dict)
    predicted_rating_target_test = classify_app_reviews(classifier, rating_tfidf_train_data, rating_target_train, rating_tfidf_test_data)
    precision, recall, f1_score = calculate_classifier_performance_metrics(rating_target_test, predicted_rating_target_test, pos_label=5)
    return precision, recall, f1_score


def feature_classify(feature_data, feature_target, not_feature_data, not_feature_target, feature_rating_dict, feature_senti_dict, feature_senti_pos_dict, feature_senti_neg_dict, feature_present_simple_dict, feature_past_simple_dict, feature_future_dict, feature_present_con_dict, classifier):
    feature_data_train, feature_target_train, feature_data_test, feature_target_test = prepare_data_for_binary_classification(feature_data, feature_target, not_feature_data, not_feature_target, is_feature=True)
    feature_tfidf_train_data, feature_tfidf_test_data = vectorize_data(feature_data_train, feature_data_test, feature_rating_dict, feature_senti_dict, feature_senti_pos_dict, feature_senti_neg_dict, feature_present_simple_dict, feature_past_simple_dict, feature_future_dict, feature_present_con_dict)
    predicted_feature_target_test = classify_app_reviews(classifier, feature_tfidf_train_data, feature_target_train, feature_tfidf_test_data)
    precision, recall, f1_score = calculate_classifier_performance_metrics(feature_target_test, predicted_feature_target_test, pos_label=7)
    return precision, recall, f1_score


def vectorize_data(data_train, data_test, rating_dict, senti_dict, senti_pos_dict, senti_neg_dict, present_simple_dict, past_simple_dict, future_dict, present_con_dict):
    vectorizer = TfidfVectorizer(use_idf=False, binary=True)
    tfidf_train_data = vectorizer.fit_transform(data_train)
    tfidf_test_data = vectorizer.transform(data_test)
    if use_metadata:
        train_review_rating, test_review_rating = get_review_rating_metadata(data_train, data_test, rating_dict)
        train_review_sentiment, test_review_sentiment = get_review_sentiment_metadata(data_train, data_test, senti_dict)
        train_review_present_simple_tense, test_review_present_simple_tense, train_review_past_simple_tense, test_review_past_simple_tense, train_review_future_tense, test_review_future_tense, train_review_continuous_tense, test_review_continuous_tense = get_review_tense_metadata(data_train, data_test, present_simple_dict, past_simple_dict, future_dict, present_con_dict)
        tfidf_train_data, tfidf_test_data = add_features_to_vectorized_data(tfidf_train_data, tfidf_test_data, train_review_rating, test_review_rating)
        tfidf_train_data, tfidf_test_data = add_features_to_vectorized_data(tfidf_train_data, tfidf_test_data, train_review_sentiment, test_review_sentiment)
        tfidf_train_data, tfidf_test_data = add_features_to_vectorized_data(tfidf_train_data, tfidf_test_data, train_review_present_simple_tense, test_review_present_simple_tense)
        tfidf_train_data, tfidf_test_data = add_features_to_vectorized_data(tfidf_train_data, tfidf_test_data, train_review_past_simple_tense, test_review_past_simple_tense)
        tfidf_train_data, tfidf_test_data = add_features_to_vectorized_data(tfidf_train_data, tfidf_test_data, train_review_future_tense, test_review_future_tense)
        tfidf_train_data, tfidf_test_data = add_features_to_vectorized_data(tfidf_train_data, tfidf_test_data, train_review_continuous_tense, test_review_continuous_tense)
    return tfidf_train_data, tfidf_test_data


def classify_app_reviews(algorithm, tfidf_train_data, target_train, tfidf_test_data):
    if algorithm == 'MultinomialNB':
        classifier = MultinomialNB()
    if algorithm == 'Tree':
        classifier = tree.DecisionTreeClassifier()
    if algorithm == 'LogisticRegression':
        classifier = LogisticRegression()
    classifier.fit(tfidf_train_data, target_train)
    predicted_target_test = classifier.predict(tfidf_test_data)
    return predicted_target_test


def calculate_classifier_performance_metrics(target_test, predicted_target_test, pos_label, binary=True):
    if binary:
        precision = metrics.precision_score(target_test, predicted_target_test, pos_label=pos_label)
        recall = metrics.recall_score(target_test, predicted_target_test, pos_label=pos_label)
        f1_score = metrics.f1_score(target_test, predicted_target_test, pos_label=pos_label)
    else:
        precision = metrics.precision_score(target_test, predicted_target_test, average=None)
        recall = metrics.recall_score(target_test, predicted_target_test, average=None)
        f1_score = metrics.f1_score(target_test, predicted_target_test, average=None)
    return precision, recall, f1_score


def prepare_data_for_binary_classification(data, target, not_data, not_target, is_feature):
    data_train = data[:feature_split_size if is_feature else split_size] + not_data[:feature_split_size if is_feature else split_size]
    target_train = target[:feature_split_size if is_feature else split_size] + not_target[:feature_split_size if is_feature else split_size]
    data_test = data[feature_split_size if is_feature else split_size:] + not_data[feature_split_size if is_feature else split_size:]
    target_test = target[feature_split_size if is_feature else split_size:] + not_target[feature_split_size if is_feature else split_size:]
    return data_train, target_train, data_test, target_test


def multiclass_classify(bugs_data, bugs_target, user_experience_data, user_experience_target, rating_data, rating_target, feature_data, feature_target, rating_dict, senti_dict, senti_pos_dict, senti_neg_dict, present_simple_dict, past_simple_dict, future_dict, present_con_dict, classifier):
    data_train, target_train, data_test, target_test = prepare_data_for_multiclass_classification(bugs_data, bugs_target, user_experience_data, user_experience_target, rating_data, rating_target, feature_data, feature_target)
    tfidf_train_data, tfidf_test_data = vectorize_data(data_train, data_test, rating_dict, senti_dict, senti_pos_dict, senti_neg_dict, present_simple_dict, past_simple_dict, future_dict, present_con_dict)
    predicted_target_test = classify_app_reviews(classifier, tfidf_train_data, target_train, tfidf_test_data)
    precision, recall, f1_score = calculate_classifier_performance_metrics(target_test, predicted_target_test, pos_label=7, binary=False)
    return precision, recall, f1_score


def prepare_data_for_multiclass_classification(bugs_data, bugs_target, user_experience_data, user_experience_target, rating_data, rating_target, feature_data, feature_target):
    data_train = bugs_data[:split_size] + user_experience_data[:split_size] + rating_data[:split_size] + feature_data[:feature_split_size]
    target_train = bugs_target[:split_size] + user_experience_target[:split_size] + rating_target[:split_size] + feature_target[:feature_split_size]
    data_test = bugs_data[split_size:] + user_experience_data[split_size:] + rating_data[split_size:] + feature_data[split_size:]
    target_test = bugs_target[split_size:] + user_experience_target[split_size:] + rating_target[split_size:] + feature_target[split_size:]
    return data_train, target_train, data_test, target_test


def get_review_rating_metadata(data_train, data_test, rating_dict):
    train_review_rating = []
    test_review_rating = []
    for i in data_train:
        train_review_rating.append([rating_dict.get(i)])
    for i in data_test:
        test_review_rating.append([rating_dict.get(i)])
    return train_review_rating, test_review_rating


def get_review_sentiment_metadata(data_train, data_test, senti_dict):
    train_review_sentiment = []
    test_review_sentiment = []
    for i in data_train:
        train_review_sentiment.append([senti_dict.get(i)])
    for i in data_test:
        test_review_sentiment.append([senti_dict.get(i)])
    return train_review_sentiment, test_review_sentiment


def get_review_2x_sentiment_metadata(data_train, data_test, senti_pos_dict, senti_neg_dict):
    train_review_positive_sentiment = []
    test_review_positive_sentiment = []
    train_review_negative_sentiment = []
    test_review_negative_sentiment = []
    for i in data_train:
        train_review_positive_sentiment.append([senti_pos_dict.get(i)])
        train_review_negative_sentiment.append([senti_neg_dict.get(i)])
    for i in data_test:
        test_review_positive_sentiment.append([senti_pos_dict.get(i)])
        test_review_negative_sentiment.append([senti_neg_dict.get(i)])
    return train_review_positive_sentiment, test_review_positive_sentiment, train_review_negative_sentiment, test_review_negative_sentiment


def get_review_tense_metadata(data_train, data_test, present_simple_dict, past_simple_dict, future_dict, present_con_dict):
    train_review_present_simple_tense = []
    test_review_present_simple_tense = []
    train_review_past_simple_tense = []
    test_review_past_simple_tense = []
    train_review_future_tense = []
    test_review_future_tense = []
    train_review_continuous_tense = []
    test_review_continuous_tense = []
    for i in data_train:
        train_review_present_simple_tense.append([present_simple_dict.get(i)])
        train_review_past_simple_tense.append([past_simple_dict.get(i)])
        train_review_future_tense.append([future_dict.get(i)])
        train_review_continuous_tense.append([present_con_dict.get(i)])
    for i in data_test:
        test_review_present_simple_tense.append([present_simple_dict.get(i)])
        test_review_past_simple_tense.append([past_simple_dict.get(i)])
        test_review_future_tense.append([future_dict.get(i)])
        test_review_continuous_tense.append([present_con_dict.get(i)])
    return train_review_present_simple_tense, test_review_present_simple_tense, train_review_past_simple_tense, test_review_past_simple_tense, train_review_future_tense, test_review_future_tense, train_review_continuous_tense, test_review_continuous_tense


def add_features_to_vectorized_data(tfidf_train_data, tfidf_test_data, train_feature, test_feature):
    tfidf_train_data = hstack((tfidf_train_data, np.array(train_feature)))
    tfidf_test_data = hstack((tfidf_test_data, np.array(test_feature)))
    return tfidf_train_data, tfidf_test_data


def split_data(train_data, cls, not_cls):

    data = []
    target = []

    not_data = []
    not_target = []

    for x in train_data:
        if x[1] == cls and cls == 'bug':
            data.append(x[0])
            target.append(1)
        if x[1] == not_cls and not_cls == 'Not_Bug_Report':
            not_data.append(x[0])
            not_target.append(0)

        if x[1] == cls and cls == 'user_experience':
            data.append(x[0])
            target.append(3)
        if x[1] == not_cls and not_cls == 'not_user_experience':
            not_data.append(x[0])
            not_target.append(2)

        if x[1] == cls and cls == 'rating':
            data.append(x[0])
            target.append(5)
        if x[1] == not_cls and not_cls == 'not_rating':
            not_data.append(x[0])
            not_target.append(4)

        if x[1] == cls and cls == 'feature':
            data.append(x[0])
            target.append(7)
        if x[1] == not_cls and not_cls == 'not_feature':
            not_data.append(x[0])
            not_target.append(6)

    return data, target, not_data, not_target

def main():
    bug = classify()

if __name__ == '__main__':
    main()