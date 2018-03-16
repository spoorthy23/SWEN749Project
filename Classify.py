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

        number_of_classes = 2

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


def main():
    bug = classify()

if __name__ == '__main__':
    main()