import pickle
import sqlite3
import random
import numpy as np
import pandas as pd
import sklearn
import sys
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class Data_Parser(object):
    '''
    数据解析
    1.对玩家的操作动作序列按天分割，以分析首留玩家，次留玩家，三留玩家
    2.对分割之后的动作序列进行tfidf特征提取
    '''
    def __init__(self, sql_file):
        self.sql_in = sql_file
        self.fc_user_ops = {}
        self.fc_user_label = {}
        self.sc_user_ops = {}
        self.sc_user_label = {}
        self.tc_user_ops = {}
        self.tc_user_label = {}
        pass

    def parse(self):        
        conn = sqlite3.connect(self.sql_in)
        c = conn.cursor()
        query_sql = "SELECT user_id, op, current_day, num_days_played, relative_timestamp \
            FROM maidian ORDER BY user_id, relative_timestamp ASC"
        for row in c.execute(query_sql):
            user_id = row[0]
            op = row[1].strip().replace(" ", '') 
            current_day = row[2]
            num_days_played = row[3]

            if current_day == 1:
                self.fc_user_label[user_id] = 1 if num_days_played == 1 else 0
                if user_id not in self.fc_user_ops:
                    self.fc_user_ops[user_id] = []
                self.fc_user_ops[user_id].append(op)
            elif current_day == 2:
                self.sc_user_label[user_id] = 1 if num_days_played == 2 else 0
                if user_id not in self.sc_user_ops:
                    self.sc_user_ops[user_id] = []
                self.sc_user_ops[user_id].append(op)
            elif current_day == 3:
                self.tc_user_label[user_id] = 1 if num_days_played == 3 else 0
                if user_id not in self.tc_user_ops:
                    self.tc_user_ops[user_id] = []
                self.tc_user_ops[user_id].append(op)
            else:
                pass
            pass

    def write_in(self, file_out):
        with open(file_out[0], 'a') as f_fc_train, open(file_out[1], 'a') as f_sc_train, open(file_out[2], 'a') as f_tc_train, \
                open(file_out[3], 'wb') as f_fc_label, open(file_out[4], 'wb') as f_sc_label, open(file_out[5], 'wb') as f_tc_label:
            fc_labels = []
            sc_labels = []
            tc_labels = []
            for user in self.fc_user_ops:
                s = ' '.join(self.fc_user_ops[user])
                f_fc_train.write(s + '\n')
                fc_labels.append(self.fc_user_label[user])
            for user in self.sc_user_ops:
                s = ' '.join(self.sc_user_ops[user])
                f_sc_train.write(s + '\n')
                sc_labels.append(self.sc_user_label[user])
            for user in self.tc_user_ops:
                s = ' '.join(self.tc_user_ops[user])
                f_tc_train.write(s + '\n')
                tc_labels.append(self.tc_user_label[user])

            pickle.dump(fc_labels, f_fc_label)
            pickle.dump(sc_labels, f_sc_label)
            pickle.dump(tc_labels, f_tc_label)

    def load_tfidf(self, *file_in, minimum_support=5, sample_rate=0, method='tfidf'):
        '''
        对得到的数据进行tf-idf特征处理
        Args:
            file_in: training data and label data
            minimum_support (int): minimum count for op
            sample rate (int [0, 10)): if 0 not sample
            method (string): can either be 'count' or 'tfidf'
        Returns:
            X: (np.array)
            Y: (list)
            op: (list)
        '''
        assert len(file_in) == 2
        corpus = []
        new_corpus = []
        new_labels = []
        op_counts = {}

        with open(file_in[0], 'rb') as f_train:
            for line in f_train:
                ops = set(line.decode('utf-8').strip().split(' '))
                for op in ops:
                    if op not in op_counts:
                        op_counts[op] = 1
                    else:
                        op_counts[op] += 1

        with open(file_in[0], 'rb') as f_train, open(file_in[1], 'rb') as f_label:
            for line in f_train:
                ops = line.decode('utf-8').strip().split(' ')
                [ops.remove(op)
                 for op in ops if op_counts[op] <= minimum_support]
                line = ' '.join(ops)
                corpus.append(line)

            labels = pickle.load(f_label)

        if sample_rate != 0:
            sampled_corpus = []
            sampled_labels = []
            sample_index = []
            for i in range(len(labels)):
                if labels[i] == 0:
                    sampled_corpus.append(corpus[i])
                    sampled_labels.append(labels[i])
                else:
                    if random.randint(0, 100) > sample_rate * 100:
                        sampled_corpus.append(corpus[i])
                        sampled_labels.append(labels[i])
            new_corpus = sampled_corpus
            new_labels = sampled_labels
        else:
            new_corpus = corpus
            new_labels = labels
            pass

        vectorizer = CountVectorizer(analyzer=str.split)

        if method.lower() == 'count':
            X = vectorizer.fit_transform(
                new_corpus).toarray()  # 该步骤仅仅取得了动作的计数而不是比值的大小
        elif method.lower() == 'tfidf':
            transformer = TfidfTransformer()
            tfidf = transformer.fit_transform(
                vectorizer.fit_transform(new_corpus))
            X = tfidf.toarray()
        Y = new_labels
        op = vectorizer.get_feature_names()
        
        # 此时需要对
        return X, Y, op
    
