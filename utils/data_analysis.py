import pickle
import sqlite3
import random
import numpy as np
import pandas as pd
import sklearn
import sys
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class Data_Analysis():
    '''
    data analysis for churn rate, interval of each op
    the pearson between interval and churn rate 
    and so on 
    '''

    def __init__(self, *file_in):
        '''
        Args:
            file_in: ops path (file), label path (pickle), sql file path (db file) 
        '''
        self.file_ops = file_in[0]
        self.file_labels = file_in[1]
        self.sql_in = file_in[2]
        with open(file_in[0], 'rb') as f_ops, open(file_in[1], 'rb') as f_label:
            self.user_labels = pickle.load(f_label)  # get a label list 1 for churned 0 for not churned 
        self.op_categories = set()
        self.op_churn = {}
        self.op_intervals = {}
        self.op_median_intervals = {}
        self.op_clicks = {}
        self.op_stage = {}
        with open(self.file_ops, 'rb') as f_ops:
            for ops in f_ops:
                ops = ops.decode('utf-8')
                ops_list = ops.strip().split(' ')
                [self.op_categories.add(op) for op in ops_list]

    def statistics_op_churn(self):
        '''
        Return:
            (dict): key (string): the op's name, 
            value (list, length is 2) 
            [0] for how many user not churned when did this op
            [1] for how many user churned when did this op
        '''
        with open(self.file_ops, 'rb') as f_ops:
            i = 0
            for ops in f_ops:
                ops = ops.decode('utf-8')
                ops_list = ops.strip().split(' ')
                if self.user_labels[i] == 0:
                    # how many user not churned at this op
                    for op in set(ops_list):
                        if op not in self.op_churn:
                            self.op_churn[op] = [0, 0]
                        self.op_churn[op][0] += 1
                else:
                    # how many user churned at this op
                    for op in set(ops_list[:-1]):
                        if op not in self.op_churn:
                            self.op_churn[op] = [0, 0]
                        self.op_churn[op][0] += 1
                    if ops_list[-1] not in self.op_churn:
                        self.op_churn[ops_list[-1]] = [0, 0]
                    self.op_churn[ops_list[-1]][1] += 1
                i += 1
        return self.op_churn

    def statistics_op_clicks(self):
        '''        
        Return:
            (dict): key(string): op's name 
            value (float list of length of 2):
            [0] for the average clicks of the unchurned user,
            [1] for the average clicks of the churned user

        '''
        sum_users = [0] * 2
        with open(self.file_ops, 'rb') as f_ops:
            i = 0
            for ops in f_ops:
                sum_users[self.user_labels[i]] += 1
                ops = ops.decode('utf-8')
                ops_list = ops.strip().split(' ')
                for op in ops_list:
                    if op not in self.op_clicks:
                        self.op_clicks[op] = [0, 0]
                    self.op_clicks[op][self.user_labels[i]] += 1
                i += 1
        for k, v in self.op_clicks.items():
            a = v[0] * 1.0 / sum_users[0]
            b = v[1] * 1.0 / sum_users[1]
            self.op_clicks[k] = [a, b]
        return self.op_clicks

    def statistics_op_avg_clicks_ratio(self):
        '''
        Return:
            (float): ratio the total clilcks of unchurn user / the total clicks of churn user 
        '''
        total_clicks = [[], []]
        with open(self.file_ops, 'rb') as f_ops:
            i = 0
            for ops in f_ops:
                ops = ops.decode('utf-8')
                ops_list = ops.strip().split(' ')
                total_clicks[self.user_labels[i]].append(len(ops_list))
                i += 1
        total_clicks[0].remove(max(total_clicks[0]))
        total_clicks[0].remove(min(total_clicks[0]))
        total_clicks[1].remove(max(total_clicks[1]))
        total_clicks[1].remove(min(total_clicks[1]))
        return np.mean(total_clicks[0]) * 1.0 / np.mean(total_clicks[1])

    def statistics_op_intervals(self):
        '''
        Returns:
            (dict): key (string): op's name, value (float): the mean interval of this op 
            (dict): key (string): op's name, value (float): the median interval of this op  
        '''
        conn = sqlite3.connect(self.sql_in)
        c = conn.cursor()
        query_sql = "SELECT user_id, op, current_day, num_days_played, relative_timestamp \
            FROM maidian ORDER BY user_id, relative_timestamp"

        previous_relativetimestamp = 0
        previous_userid = None
        previous_op = None
        intervals = []
        for row in c.execute(query_sql):
            user_id = row[0]
            op = row[1].strip().replace(' ', '')
            current_day = row[2]
            num_days_played = row[3]
            relative_timestamp = row[4]
            # calculate the interval
            interval = relative_timestamp - previous_relativetimestamp
            if previous_userid == user_id:
                if previous_op not in self.op_intervals:
                    self.op_intervals[previous_op] = []
                self.op_intervals[previous_op].append(interval)
            else:
                pass
            previous_userid = user_id
            previous_relativetimestamp = relative_timestamp
            previous_op = op

        for k, intervals in self.op_intervals.items():
            self.op_median_intervals[k] = np.median(intervals)
            if len(intervals) >= 10:
                intervals.remove(max(intervals))
                intervals.remove(max(intervals))
                intervals.remove(min(intervals))
                intervals.remove(min(intervals))
            self.op_intervals[k] = np.mean(intervals)
        return self.op_intervals, self.op_median_intervals

    def statistics_op_avg_intervals(self):
        '''
        Return:
            (float): the mean of all the ops 
        '''
        intervals = []
        for _, interval in self.op_intervals.items():
            intervals.append(interval)
        intervals.remove(max(intervals))
        intervals.remove(min(intervals))
        return np.mean(intervals)

    def statistics_op_median_intervals(self):
        '''
        Return:
            (float): the median of all the ops 
        '''
        intervals = []
        for _, interval in self.op_intervals.items():
            intervals.append(interval)
        return np.median(intervals)

    def statistics_op_stage(self):
        '''
        Return:
            (dict): key (string): op's name 
                value (float list): op's first occurrences 
        '''
        conn = sqlite3.connect(self.sql_in)
        c = conn.cursor()
        query_sql = "SELECT user_id, op, relative_timestamp \
            FROM maidian ORDER BY user_id, relative_timestamp ASC"

        previous_userid = None
        start_time = None

        for row in c.execute(query_sql):
            user_id = row[0]
            op = row[1].strip().replace(' ', '')
            relative_timestamp = row[2]

            if previous_userid is None:
                start_time = relative_timestamp
                temp_dict = {}
                temp_dict[op] = relative_timestamp
            elif previous_userid == user_id:
                # only the first occurrence of the ops will be record
                if op not in temp_dict:
                    temp_dict[op] = relative_timestamp
            else:
                # the user changed                 
                for op, rt in temp_dict.items():
                    if op not in self.op_stage:
                        self.op_stage[op] = []
                    else:
                        self.op_stage[op].append(rt - start_time)
                temp_dict = {}
                start_time = relative_timestamp
                temp_dict[op] = relative_timestamp

            previous_userid = user_id
        return self.op_stage

    def statistics_pearson_clicks_intervals(self):
        '''
        Return:
            (float): the Pearson between the op's interval and the churn rate of this op
        '''
        churn_rates = []
        intervals = []
        for k, v in self.op_churn.items():
            churn_rate = v[1] * 1.0 / v[0]
            churn_rates.append(churn_rate)
            intervals.append(self.op_intervals[k])

        s1 = pd.Series(intervals)
        s2 = pd.Series(churn_rates)
        return s1.corr(s2)

    def statistics_pearson_clicks_stage(self):
        '''
        deprecated 
        used for test
        '''
        stages = []
        clicks = []

        for k, _ in self.op_clicks.items():            
            if self.op_clicks[k][1] == 0:
                clicks.append(-1)
            else:
                clicks.append(self.op_clicks[k][0]
                              * 1.0 / self.op_clicks[k][1])

            v = self.op_stage[k]
            if len(v) > 4:
                v.remove(max(v))
                v.remove(min(v))
            stages.append(np.mean(v) * 10)

        s1 = pd.Series(stages)
        s2 = pd.Series(clicks)
        return s1.corr(s2)
