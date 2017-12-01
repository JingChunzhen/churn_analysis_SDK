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
    对数据库中玩家的动作信息进行分析
    1.动作随后的时间间隔
    2.动作的留存比， 即点击该动作流失的玩家数量与点击过该动作的玩家的数量的比值
    3.动作的非流失玩家平均点击次数， 流失玩家平均点击次数
    4.非流失玩家平均点击次数，流失玩家平均点击次数
    5.动作属于游戏前期还是后期进行
    6.动作随后时间间隔与动作留存比之间的pearson系数
    '''

    def __init__(self, *file_in):
        '''
        user_ops: 为玩家的动作序列
        user_labels: 为玩家的标签
        '''
        self.file_ops = file_in[0]
        self.file_labels = file_in[1]
        self.sql_in = file_in[2]
        with open(file_in[0], 'rb') as f_ops, open(file_in[1], 'rb') as f_label:
            self.user_labels = pickle.load(f_label)  # 得到的是一个玩家的标签列表
        self.op_categories = set()
        self.op_churn = {}
        self.op_intervals = {}
        self.op_clicks = {}
        self.op_stage = {}
        with open(self.file_ops, 'rb') as f_ops:
            for ops in f_ops:
                ops = ops.decode('utf-8')
                ops_list = ops.strip().split(' ')
                [self.op_categories.add(op) for op in ops_list]

    @property
    def statistics_op_churn(self):
        '''
        计算动作留存比，即点击该动作之后流失的人数点击过该动作的人数的比值
        '''
        with open(self.file_ops, 'rb') as f_ops:
            i = 0
            for ops in f_ops:
                ops = ops.decode('utf-8')
                ops_list = ops.strip().split(' ')
                if self.user_labels[i] == 0:
                    # 多少个玩家做了这个动作而没有流失
                    for op in set(ops_list):
                        if op not in self.op_churn:
                            self.op_churn[op] = [0, 0]
                        self.op_churn[op][0] += 1
                else:
                    # 多少个玩家做了这个动作而流失
                    for op in set(ops_list[:-1]):
                        if op not in self.op_churn:
                            self.op_churn[op] = [0, 0]
                        self.op_churn[op][0] += 1
                    if ops_list[-1] not in self.op_churn:
                        self.op_churn[ops_list[-1]] = [0, 0]
                    self.op_churn[ops_list[-1]][1] += 1
                i += 1
        return self.op_churn

    @property
    def statistics_op_clicks(self):
        '''
        计算每一个动作，非流失玩家的平均点击次数和流失玩家的平均点击次数之间的比值
        还需要计算的是所有动作中非流失玩家的平均点击次数和流失玩家的平均点击次数之间的比值
        # BUG 
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

    @property
    def statistics_op_avg_clicks_ratio(self):
        '''
        Return:
            (float): 非流失玩家点击次数和流失玩家点击次数的比值
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

    @property
    def statistics_op_intervals(self):
        '''
        卡点分析， 计算动作随后的时间间隔
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
            # 计算时间间隔
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
            if len(intervals) >= 10:
                intervals.remove(max(intervals))
                intervals.remove(max(intervals))
                intervals.remove(min(intervals))
                intervals.remove(min(intervals))
            self.op_intervals[k] = np.mean(intervals)
        return self.op_intervals

    @property
    def statistics_op_avg_intervals(self):
        '''
        Returns:
            (float): 返回所有动作的平均时间间隔
        '''
        intervals = []
        for _, interval in self.op_intervals.items():
            intervals.append(interval)
        intervals.remove(max(intervals))
        intervals.remove(min(intervals))
        return np.mean(intervals)

    @property
    def statistics_op_stage(self):
        '''
        用来判断一个动作是游戏的前期还是后期动作
        '''
        conn = sqlite3.connect(self.sql_in)
        c = conn.cursor()
        query_sql = "SELECT user_id, op, relative_timestamp \
            FROM maidian ORDER BY user_id, relative_timestamp ASC"

        previous_userid = None
        start_time = None
        end_time = None
        previous_relativetimestamp = None
        for row in c.execute(query_sql):
            user_id = row[0]
            op = row[1].strip().replace(' ', '')
            relative_timestamp = row[2]

            if previous_userid is None:
                start_time = relative_timestamp
                temp_dict = {}
                temp_dict[op] = relative_timestamp
            elif previous_userid == user_id:
                if op not in temp_dict:
                    temp_dict[op] = relative_timestamp
            else:
                # 此时换了一个玩家
                end_time = previous_relativetimestamp
                for op, rt in temp_dict.items():
                    if op not in self.op_stage:
                        self.op_stage[op] = []
                    if end_time - start_time == 0:
                        pass
                    else:
                        self.op_stage[op].append(
                            (rt - start_time) * 1.0 / (end_time - start_time))
                temp_dict = {}
                start_time = relative_timestamp
                temp_dict[op] = relative_timestamp

            previous_relativetimestamp = relative_timestamp
            previous_userid = user_id
        return self.op_stage

    @property
    def statistics_pearson_clicks_intervals(self):
        '''
        计算动作留存比和动作随后时间间隔之间的pearson系数
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

    @property
    def statistics_pearson_clicks_stage(self):
        '''
        计算非流失玩家和流失玩家点击次数的比值和动作阶段之间的pearson系数  
        在temp文件中的玩家的动作序列是按照天分割之后的，而数据库中的文件涵盖了很多天的数据
        所以二者中的动作种类数量是不一样的      
        '''
        stages = []
        clicks = []        

        for k, _ in self.op_clicks.items():
            '''
            为确保bug不出现，需要先遍历op_clicks而不是op_stage
            因为op_stage是由所有数据库文件中的数据生成
            '''
            if self.op_clicks[k][1] == 0:                
                clicks.append(-1)
            else:
                clicks.append(self.op_clicks[k][0] * 1.0 / self.op_clicks[k][1])

            v = self.op_stage[k]
            if len(v) > 4:
                v.remove(max(v))
                v.remove(min(v))
            stages.append(np.mean(v) * 10)
                        
        s1 = pd.Series(stages)
        s2 = pd.Series(clicks)
        return s1.corr(s2)

    def draw(self):
        from pyecharts import Line
        line = Line("")
        churn_rates = []
        intervals = []
        for k, v in self.op_churn.items():
            churn_rate = v[1] * 1.0 / v[0]
            churn_rates.append(churn_rate)
            intervals.append(self.op_intervals[k])
        attr = [_ for _ in range(len(intervals))]
        churnrates = [rate * 1000 for rate in churn_rates]

        new_intervals = [i for i in intervals[::10]]
        new_churnrates = [c for c in churnrates[::10]]
        attr = [i for i in range(0, len(intervals), 10)]
        line = Line("")
        line.add("动作随后时间间隔", attr, new_intervals)
        line.add("动作留存比 * 1000", attr, new_churnrates)
        line.show_config()
        line.render()
