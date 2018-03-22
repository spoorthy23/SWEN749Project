def fetch_bug_data(self, nlp_type):
    cls = 'bug'
    not_cls = 'Not_Bug_Report'

    train = []

    rating_dict = {}
    senti_dict = {}
    senti_pos_dict = {}
    senti_neg_dict = {}
    present_simple_dict = {}
    past_simple_dict = {}
    future_dict = {}
    present_con_dict = {}

    self.cur.execute("SELECT * FROM bug_report_data_train")

    for row in self.cur.fetchall():

        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})

        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})

        train.append((decoded_row, cls))

    self.cur.execute("SELECT * FROM Not_Bug_Report_Data_Train")

    for row in self.cur.fetchall():
        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})
        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})

        train.append((decoded_row, not_cls))

    self.cur.execute("SELECT * FROM Bug_Report_Data_Test")

    for row in self.cur.fetchall():
        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})
        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})
        decoded_row = str(decoded_row)
        train.append((decoded_row, cls))

    self.cur.execute("SELECT * FROM Not_Bug_Report_Data_Test")

    for row in self.cur.fetchall():
        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})
        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})
        decoded_row = str(decoded_row)
        decoded_row_1 = str(decoded_row_1)

        train.append((decoded_row, not_cls))

    self.db.commit()
    return train, rating_dict, senti_dict, senti_pos_dict, senti_neg_dict, present_simple_dict, past_simple_dict, future_dict, present_con_dict, cls, not_cls


def fetch_user_experience_data(self, nlp_type):
    cls = 'user_experience'
    not_cls = 'not_user_experience'

    train = []

    rating_dict = {}
    senti_dict = {}
    senti_pos_dict = {}
    senti_neg_dict = {}
    present_simple_dict = {}
    past_simple_dict = {}
    future_dict = {}
    present_con_dict = {}

    self.cur.execute("SELECT * FROM userexperience_data_train")

    for row in self.cur.fetchall():

        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})

        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})

        train.append((decoded_row, cls))

    self.cur.execute("SELECT * FROM not_userexperience_data_train")

    for row in self.cur.fetchall():
        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})
        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})

        train.append((decoded_row, not_cls))

    self.cur.execute("SELECT * FROM userexperience_data_test")

    for row in self.cur.fetchall():
        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})
        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})
        decoded_row = str(decoded_row)
        train.append((decoded_row, cls))

    self.cur.execute("SELECT * FROM not_userexperience_data_test")

    for row in self.cur.fetchall():
        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})
        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})
        decoded_row = str(decoded_row)
        decoded_row_1 = str(decoded_row_1)

        train.append((decoded_row, not_cls))

    self.db.commit()
    return train, rating_dict, senti_dict, senti_pos_dict, senti_neg_dict, present_simple_dict, past_simple_dict, future_dict, present_con_dict, cls, not_cls


def fetch_rating_data(self, nlp_type):
    cls = 'rating'
    not_cls = 'not_rating'
    train = []

    rating_dict = {}
    senti_dict = {}
    senti_pos_dict = {}
    senti_neg_dict = {}
    present_simple_dict = {}
    past_simple_dict = {}
    future_dict = {}
    present_con_dict = {}

    self.cur.execute("SELECT * FROM rating_data_train")

    for row in self.cur.fetchall():

        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})

        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})

        train.append((decoded_row, cls))

    self.cur.execute("SELECT * FROM not_rating_data_train")

    for row in self.cur.fetchall():
        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})
        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})

        train.append((decoded_row, not_cls))

    self.cur.execute("SELECT * FROM rating_data_test")

    for row in self.cur.fetchall():
        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})
        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})
        decoded_row = str(decoded_row)
        train.append((decoded_row, cls))

    self.cur.execute("SELECT * FROM not_rating_data_test")

    for row in self.cur.fetchall():
        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})
        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})
        decoded_row = str(decoded_row)
        decoded_row_1 = str(decoded_row_1)

        train.append((decoded_row, not_cls))

    self.db.commit()
    return train, rating_dict, senti_dict, senti_pos_dict, senti_neg_dict, present_simple_dict, past_simple_dict, future_dict, present_con_dict, cls, not_cls


def fetch_feature_data(self, nlp_type):
    cls = 'feature'
    not_cls = 'not_feature'

    train = []

    rating_dict = {}
    senti_dict = {}
    senti_pos_dict = {}
    senti_neg_dict = {}
    present_simple_dict = {}
    past_simple_dict = {}
    future_dict = {}
    present_con_dict = {}

    self.cur.execute("SELECT * FROM feature_or_improvment_request_data_train")

    for row in self.cur.fetchall():

        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})

        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})

        train.append((decoded_row, cls))

    self.cur.execute("SELECT * FROM not_feature_or_improvment_request_data_train")

    for row in self.cur.fetchall():
        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})
        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})

        train.append((decoded_row, not_cls))

    self.cur.execute("SELECT * FROM feature_or_improvment_request_data_test")

    for row in self.cur.fetchall():
        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})
        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})
        decoded_row = str(decoded_row)
        train.append((decoded_row, cls))

    self.cur.execute("SELECT * FROM not_feature_or_improvment_request_data_test")

    for row in self.cur.fetchall():
        if nlp_type == 'stopwords_removal_lemmatization':
            decoded_row = str(row[17])  # 'stopwords_removal_lemmatization'
        if nlp_type == 'stopwords_removal':
            decoded_row = str(row[10])  # 'stopwords_removal'
        if nlp_type == 'lemmatized_comment':
            decoded_row = str(row[11])  # 'lemmatized_comment'
        if nlp_type == 'original_comment':
            decoded_row = str(row[4])  # 'original_comment'

        decoded_row_1 = str(row[2])  # 'reviewId'
        rating = row[5]
        sentiScore = row[13]
        senti_pos = row[14]
        senti_neg = row[15]
        present_simple = row[19]
        present_con = row[20]
        past_simple = row[21]
        future = row[22]
        if present_simple:
            present_simple = float(int(present_simple))
            present_simple_dict.update({decoded_row: present_simple})
        if present_con:
            present_con = float(int(present_con))
            present_con_dict.update({decoded_row: present_con})
        if past_simple:
            past_simple = float(int(past_simple))
            past_simple_dict.update({decoded_row: past_simple})
        if future:
            future = float(int(future))
            future_dict.update({decoded_row: future})
        if rating:
            rating = float(rating)
            rating = int(rating)
            rating_dict.update({decoded_row: rating})
            rating_dict.update({decoded_row_1: rating})
        senti_dict.update({decoded_row: sentiScore})
        senti_pos_dict.update({decoded_row: senti_pos})
        senti_neg_dict.update({decoded_row: senti_neg})
        decoded_row = str(decoded_row)
        decoded_row_1 = str(decoded_row_1)

        train.append((decoded_row, not_cls))

    self.db.commit()
    return train, rating_dict, senti_dict, senti_pos_dict, senti_neg_dict, present_simple_dict, past_simple_dict, future_dict, present_con_dict, cls, not_cls