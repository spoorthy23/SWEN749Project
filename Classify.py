from database import fetch_bug_data
from database import fetch_feature_data
from database import fetch_rating_data
from database import fetch_user_experience_data

import MySQLdb

class classify:
    def __init__(self):
        db = MySQLdb.connect(host="127.0.0.1", user="root", passwd="student", db="sys")

        db.autocommit(True)
        db.begin()
        cur = db.cursor()
        self.db = db
        self.cur = cur

        self.bug_train, self.bug_rating_dict, self.bug_senti_dict, self.bug_senti_pos_dict, self.bug_senti_neg_dict, self.bug_present_simple_dict, self.bug_past_simple_dict, self.bug_future_dict, self.bug_present_con_dict, self.bug_cls, self.not_bug_cls = fetch_bug_data(
            self)
        self.feature_train, self.feature_rating_dict, self.feature_senti_dict, self._senti_pos_dict, self.feature_senti_neg_dict, self.feature_present_simple_dict, self.feature_past_simple_dict, self.feature_future_dict, self.feature_present_con_dict, self.feature_cls, self.not_feature_cls = fetch_feature_data(
            self)
        self.rating_train, self.rating_rating_dict, self.rating_senti_dict, self.rating_senti_pos_dict, self.rating_senti_neg_dict, self.rating_present_simple_dict, self.rating_past_simple_dict, self.rating_future_dict, self.rating_present_con_dict, self.rating_cls, self.not_rating_cls = fetch_rating_data(
            self)
        self.user_experience_train, self.user_experience_rating_dict, self.user_experience_senti_dict, self.user_experience_senti_pos_dict, self.user_experience_senti_neg_dict, self.user_experience_present_simple_dict, self.user_experience_past_simple_dict, self.user_experience_future_dict, self.user_experience_present_con_dict, self.user_experience_cls, self.not_user_experience_cls = fetch_user_experience_data(
            self)

        self.db.commit()
        self.db.close()

        split_size = 260
        feature_split_size = 207

        bugs_data, bugs_target, not_bugs_data, not_bugs_target = split_data(self.bug_train, self.bug_cls)
        user_experience_data, user_experience_target, not_user_experience_data, not_user_experience_target = split_data(self.user_experience_train, self.user_experience_cls)
        rating_data, rating_target, not_rating_data, not_rating_target = split_data(self.rating_train, self.rating_cls)
        feature_data, feature_target, not_feature_data, not_feature_target = split_data(self.feature_train, self.feature_cls)

        # print(len(self.bug_train), len(bugs_data), len(bugs_target), len(not_bugs_data), len(not_bugs_target))
        # print(len(self.user_experience_train), len(user_experience_data), len(user_experience_target), len(not_user_experience_data), len(not_user_experience_target))
        # print(len(self.rating_train), len(rating_data), len(rating_target), len(not_rating_data), len(not_rating_target))
        # print(len(self.feature_train), len(feature_data), len(feature_target), len(not_feature_data), len(not_feature_target))

        bug_classify(bugs_data, bugs_target, not_bugs_data, not_bugs_target, split_size)
        user_experience_classify(user_experience_data, user_experience_target, not_user_experience_data, not_user_experience_target, split_size)
        rating_classify(rating_data, rating_target, not_rating_data, not_rating_target, split_size)
        feature_classify(feature_data, feature_target, not_feature_data, not_feature_target, feature_split_size)


def bug_classify(bugs_data, bugs_target, not_bugs_data, not_bugs_target, split_size):
    bugs_data_train = bugs_data[:split_size] + not_bugs_data[:split_size]
    bugs_target_train = bugs_target[:split_size] + not_bugs_target[:split_size]

    bugs_data_test = bugs_data[split_size:] + not_bugs_data[split_size:]
    bugs_target_test = bugs_target[split_size:] + not_bugs_target[split_size:]

    # print(len(bugs_data_train), len(bugs_target_train), len(bugs_data_test), len(bugs_target_test))


def user_experience_classify(user_experience_data, user_experience_target, not_user_experience_data, not_user_experience_target, split_size):
    user_experience_data_train = user_experience_data[:split_size] + not_user_experience_data[:split_size]
    user_experience_target_train = user_experience_target[:split_size] + not_user_experience_target[:split_size]

    user_experience_data_test = user_experience_data[split_size:] + not_user_experience_data[split_size:]
    user_experience_target_test = user_experience_target[split_size:] + not_user_experience_target[split_size:]

    # print(len(user_experience_data_train), len(user_experience_target_train), len(user_experience_data_test), len(user_experience_target_test))


def rating_classify(rating_data, rating_target, not_rating_data, not_rating_target, split_size):
    rating_data_train = rating_data[:split_size] + not_rating_data[:split_size]
    rating_target_train = rating_target[:split_size] + not_rating_target[:split_size]

    rating_data_test = rating_data[split_size:] + not_rating_data[split_size:]
    rating_target_test = rating_target[split_size:] + not_rating_target[split_size:]

    # print(len(rating_data_train), len(rating_target_train), len(rating_data_test), len(rating_target_test))


def feature_classify(feature_data, feature_target, not_feature_data, not_feature_target, feature_split_size):
    feature_data_train = feature_data[:feature_split_size] + not_feature_data[:feature_split_size]
    feature_target_train = feature_target[:feature_split_size] + not_feature_target[:feature_split_size]

    feature_data_test = feature_data[feature_split_size:] + not_feature_data[feature_split_size:]
    feature_target_test = feature_target[feature_split_size:] + not_feature_target[feature_split_size:]

    # print(len(feature_data_train), len(feature_target_train), len(feature_data_test), len(feature_target_test))


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