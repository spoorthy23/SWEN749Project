from database import fetch_bug_data
from database import fetch_feature_data
from database import fetch_rating_data
from database import fetch_user_experience_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import MySQLdb

split_size = 260
feature_split_size = 207

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

        nlp_type = original_comment

        self.bug_train, self.bug_rating_dict, self.bug_senti_dict, self.bug_senti_pos_dict, self.bug_senti_neg_dict, self.bug_present_simple_dict, self.bug_past_simple_dict, self.bug_future_dict, self.bug_present_con_dict, self.bug_cls, self.not_bug_cls = fetch_bug_data(
            self, nlp_type)
        self.feature_train, self.feature_rating_dict, self.feature_senti_dict, self._senti_pos_dict, self.feature_senti_neg_dict, self.feature_present_simple_dict, self.feature_past_simple_dict, self.feature_future_dict, self.feature_present_con_dict, self.feature_cls, self.not_feature_cls = fetch_feature_data(
            self, nlp_type)
        self.rating_train, self.rating_rating_dict, self.rating_senti_dict, self.rating_senti_pos_dict, self.rating_senti_neg_dict, self.rating_present_simple_dict, self.rating_past_simple_dict, self.rating_future_dict, self.rating_present_con_dict, self.rating_cls, self.not_rating_cls = fetch_rating_data(
            self, nlp_type)
        self.user_experience_train, self.user_experience_rating_dict, self.user_experience_senti_dict, self.user_experience_senti_pos_dict, self.user_experience_senti_neg_dict, self.user_experience_present_simple_dict, self.user_experience_past_simple_dict, self.user_experience_future_dict, self.user_experience_present_con_dict, self.user_experience_cls, self.not_user_experience_cls = fetch_user_experience_data(
            self, nlp_type)

        self.db.commit()
        self.db.close()

        bugs_data, bugs_target, not_bugs_data, not_bugs_target = split_data(self.bug_train, self.bug_cls)
        user_experience_data, user_experience_target, not_user_experience_data, not_user_experience_target = split_data(self.user_experience_train, self.user_experience_cls)
        rating_data, rating_target, not_rating_data, not_rating_target = split_data(self.rating_train, self.rating_cls)
        feature_data, feature_target, not_feature_data, not_feature_target = split_data(self.feature_train, self.feature_cls)

        bug_precision, bug_recall, bug_f1_score = bug_classify(bugs_data, bugs_target, not_bugs_data, not_bugs_target)
        print("Bug: ", bug_precision, bug_recall, bug_f1_score)
        user_experience_precision, user_experience_recall, user_experience_f1_score = user_experience_classify(user_experience_data, user_experience_target, not_user_experience_data, not_user_experience_target)
        print("user Experience: ", user_experience_precision, user_experience_recall, user_experience_f1_score)
        rating_precision, rating_recall, rating_f1_score = rating_classify(rating_data, rating_target, not_rating_data, not_rating_target)
        print("Rating: ", rating_precision, rating_recall, rating_f1_score)
        feature_precision, feature_recall, feature_f1_score = feature_classify(feature_data, feature_target, not_feature_data, not_feature_target)
        print("Feature Request: ", feature_precision, feature_recall, feature_f1_score)


def bug_classify(bugs_data, bugs_target, not_bugs_data, not_bugs_target):
    bugs_data_train, bugs_target_train, bugs_data_test, bugs_target_test = prepare_data_for_classification(bugs_data, bugs_target, not_bugs_data, not_bugs_target, is_feature=False)
    bugs_tfidf_train_data, bugs_tfidf_test_data = vectorize(bugs_data_train, bugs_data_test)
    predicted_bugs_target_test = classify_app_reviews('MultinomialNB', bugs_tfidf_train_data, bugs_target_train, bugs_tfidf_test_data)
    precision, recall, f1_score = calculate_classifier_performance_metrics(bugs_target_test, predicted_bugs_target_test)
    return precision, recall, f1_score


def user_experience_classify(user_experience_data, user_experience_target, not_user_experience_data, not_user_experience_target):
    user_experience_data_train, user_experience_target_train, user_experience_data_test, user_experience_target_test = prepare_data_for_classification(user_experience_data, user_experience_target, not_user_experience_data, not_user_experience_target, is_feature=False)
    user_experience_tfidf_train_data, user_experience_tfidf_test_data = vectorize(user_experience_data_train, user_experience_data_test)
    predicted_user_experience_target_test = classify_app_reviews('MultinomialNB', user_experience_tfidf_train_data, user_experience_target_train, user_experience_tfidf_test_data)
    precision, recall, f1_score = calculate_classifier_performance_metrics(user_experience_target_test, predicted_user_experience_target_test)
    return precision, recall, f1_score


def rating_classify(rating_data, rating_target, not_rating_data, not_rating_target):
    rating_data_train, rating_target_train, rating_data_test, rating_target_test = prepare_data_for_classification(rating_data, rating_target, not_rating_data, not_rating_target, is_feature=False)
    rating_tfidf_train_data, rating_tfidf_test_data = vectorize(rating_data_train, rating_data_test)
    predicted_rating_target_test = classify_app_reviews('MultinomialNB', rating_tfidf_train_data, rating_target_train, rating_tfidf_test_data)
    precision, recall, f1_score = calculate_classifier_performance_metrics(rating_target_test, predicted_rating_target_test)
    return precision, recall, f1_score


def feature_classify(feature_data, feature_target, not_feature_data, not_feature_target):
    feature_data_train, feature_target_train, feature_data_test, feature_target_test = prepare_data_for_classification(feature_data, feature_target, not_feature_data, not_feature_target, is_feature=True)
    feature_tfidf_train_data, feature_tfidf_test_data = vectorize(feature_data_train, feature_data_test)
    predicted_feature_target_test = classify_app_reviews('MultinomialNB', feature_tfidf_train_data, feature_target_train, feature_tfidf_test_data)
    precision, recall, f1_score = calculate_classifier_performance_metrics(feature_target_test, predicted_feature_target_test)
    return precision, recall, f1_score



def vectorize(data_train, data_test):
    vectorizer = TfidfVectorizer(use_idf=False, binary=True)
    tfidf_train_data = vectorizer.fit_transform(data_train)
    tfidf_test_data = vectorizer.transform(data_test)
    return tfidf_train_data, tfidf_test_data


def classify_app_reviews(algorithm, tfidf_train_data, target_train, tfidf_test_data):
    if algorithm == 'MultinomialNB':
        classifier = MultinomialNB()
    classifier.fit(tfidf_train_data, target_train)
    predicted_target_test = classifier.predict(tfidf_test_data)
    return predicted_target_test


def calculate_classifier_performance_metrics(target_test, predicted_target_test):
    precision = metrics.precision_score(target_test, predicted_target_test)
    recall = metrics.recall_score(target_test, predicted_target_test)
    f1_score = metrics.f1_score(target_test, predicted_target_test)
    return precision, recall, f1_score


def prepare_data_for_classification(data, target, not_data, not_target, is_feature):
    data_train = data[:feature_split_size if is_feature else split_size] + not_data[:feature_split_size if is_feature else split_size]
    target_train = target[:feature_split_size if is_feature else split_size] + not_target[:feature_split_size if is_feature else split_size]
    data_test = data[feature_split_size if is_feature else split_size:] + not_data[feature_split_size if is_feature else split_size:]
    target_test = target[feature_split_size if is_feature else split_size:] + not_target[feature_split_size if is_feature else split_size:]
    return data_train, target_train, data_test, target_test


def split_data(train_data, cls):

    data = []
    target = []

    not_data = []
    not_target = []

    for x in train_data:
        if x[1] == cls:
            data.append(x[0])
            target.append(1)
        else:
            not_data.append(x[0])
            not_target.append(0)

    return data, target, not_data, not_target

def main():
    bug = classify()

if __name__ == '__main__':
    main()