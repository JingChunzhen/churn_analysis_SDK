from utils.data_parse import Data_Parser
from utils.data_analysis import Data_Analysis
from xgb.xgb_model import XGB_Model
import os
import numpy as np
import pandas as pd
import time


class Main():
    '''
    pipeline for game churn analysis
    change the code to fit it in all the ops, not only the key ops 
    '''

    def __init__(self, day, sql_in, k):
        self.day = day
        self.sql_in = sql_in
        self.k = k
        self.training_data = {
            1: './temp/fc_train.txt',
            2: './temp/sc_train.txt',
            3: './temp/tc_train.txt'
        }
        self.training_label = {
            1: './temp/fc_label.pkl',
            2: './temp/sc_label.pkl',
            3: './temp/tc_label.pkl'
        }

        self.x = None
        self.y = None
        self.op = None

        # training data generate
        self._load_data()
        self.da = Data_Analysis(
            self.training_data[self.day], self.training_label[self.day], self.sql_in)
        self.op_churn = self.da.statistics_op_churn()
        self.op_clicks = self.da.statistics_op_clicks()
        temp_intervals = self.da.statistics_op_intervals()
        self.op_intervals = temp_intervals[0]
        self.op_median_intervals = temp_intervals[1]
        self.op_stages = self.da.statistics_op_stage()

        # new dict         
        self.op_verbose = {}
        # end of new dict 

        self.key_ops = None

    def _load_data(self):
        dp = Data_Parser(self.sql_in)
        if os.path.exists(self.training_data[self.day]) and os.path.exists(self.training_label[self.day]):
            pass
        else:
            dp.parse()
            dp.write_in([
                './temp/fc_train.txt',
                './temp/sc_train.txt',
                './temp/tc_train.txt',
                './temp/fc_label.pkl',
                './temp/sc_label.pkl',
                './temp/tc_label.pkl'
            ])
        self.x, self.y, self.op = dp.load_tfidf(
            self.training_data[self.day], self.training_label[self.day])

    def _get_key_ops(self):
        # self._load_data()
        xgb = XGB_Model(self.x, self.y, self.op, 0.2, 0.1)
        xgb.model()
        print(len(xgb.key_ops))
        self.key_ops = xgb.key_ops

    def _get_op_berbose(self):
        '''
        '''
        import xlrd        
        data = xlrd.open_workbook('./data/动作说明.xlsx')
        table = data.sheet_by_name('Sheet1')

        nrows = table.nrows
        
        for i in range(nrows):
            line = table.row_values(i)
            if line[0] not in self.op_verbose:
                self.op_verbose[line[0]] = line[1]                

    def ops_analysis(self):
        self._get_key_ops()
        self._get_op_berbose()
        sorted_keyops = np.argsort(self.key_ops)
        print(len(self.key_ops))

        st_op_churn = {}
        st_op_clicks = {}
        st_op_stage = {}
        st_op_churnnum = {}

        for k, v in self.op_churn.items():
            if v[0] == 0:
                st_op_churn[k] = -1
            else:
                st_op_churn[k] = v[1] * 1.0 / v[0]
            st_op_churnnum[k] = v[1]

        for k, v in self.op_clicks.items():
            if v[1] == 0:
                st_op_clicks[k] = -1 * v[0]
            else:
                st_op_clicks[k] = v[0] * 1.0 / v[1]

        for k, v in self.op_stages.items():
            if len(v) > 4:
                v.remove(max(v))
                v.remove(min(v))
            st_op_stage[k] = np.mean(v)
            pass

        # here
        print('动作平均时间间隔: {}'.format(self.da.statistics_op_avg_intervals()))
        print('动作平均点击比例: {}'.format(self.da.statistics_op_avg_clicks_ratio()))
        print('动作平均时间间隔和动作留存比之间的pearson系数: {}'.format(
            self.da.statistics_pearson_clicks_intervals()))
        print('动作点击比值和动作阶段之间的pearson系数: {}'.format(
            self.da.statistics_pearson_clicks_stage()))
        self.draw()

        if self.k is True:
            # 只对关键动作进行分析
            data = []
            for opid in sorted_keyops[::-1]:
                op_name = self.op[opid]
                if self.key_ops[opid] == 0:
                    break
                # 动作的名称， 动作的留存比， 动作的点击次数比值， 动作属于前期还是后期动作， 动作的平均时间间隔， 动作随后时间间隔的中位数， 动作的重要性
                verbose = self.op_verbose[op_name] if op_name in self.op_verbose else ""
                print('{}|{}|{:.5f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.4f}'.format(
                    op_name, verbose, st_op_churn[op_name], st_op_churnnum[op_name], st_op_clicks[op_name], st_op_stage[op_name], self.op_intervals[op_name], self.op_median_intervals[op_name], self.key_ops[opid]))
                data.append(
                    [op_name,
                    verbose,
                     "{:.5f}".format(st_op_churn[op_name]),
                     "{:.2f}".format(st_op_churnnum[op_name]),
                     "{:.2f}".format(st_op_clicks[op_name]),
                     "{:.2f}".format(st_op_stage[op_name]),
                     "{:.2f}".format(self.op_intervals[op_name]),
                     "{:.2f}".format(self.op_median_intervals[op_name]),
                     "{:.4f}".format(self.key_ops[opid])]
                )

            df = pd.DataFrame(data=data, columns=[
                '动作', '动作详细说明', '留存比', '动作流失人数', '点击次数比', '动作时段', '随后时间间隔', '随后中位数时间间隔', '重要性'])

            self.writeTo(parent_dir='output', path='',
                         file_name='xlsx', pd_file=df)
        else:
            # 对全部的动作进行分析
            data = []
            for op_name, _ in st_op_clicks.items():
                # 动作的名称，动作的留存比，动作的点击次数的比值，动作属于前期还是后期动作，动作的平均时间间隔，动作的中位数时间间隔
                print('{}|{:.5f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}'.format(
                    op_name, st_op_churn[op_name], st_op_churnnum[op_name], st_op_clicks[op_name], st_op_stage[op_name], self.op_intervals[op_name], self.op_median_intervals[op_name]))
                data.append(
                    [op_name,
                     "{:.5f}".format(st_op_churn[op_name]),
                     "{:.2f}".format(st_op_churnnum[op_name]),
                     "{:.2f}".format(st_op_clicks[op_name]),
                     "{:.2f}".format(st_op_stage[op_name]),
                     "{:.2f}".format(self.op_intervals[op_name]),
                     "{:.2f}".format(self.op_median_intervals[op_name])]
                )

            df = pd.DataFrame(data=data, columns=[
                '动作', '留存比', '动作流失人数', '点击次数比', '动作时段', '随后平均时间间隔', '随后中位数时间间隔'])

            self.writeTo(parent_dir='output', path='',
                         file_name='csv', pd_file=df)

    def writeTo(self, parent_dir, path, file_name, pd_file, output_format='csv'):
        full_path = os.path.join(parent_dir, path)
        os.makedirs(full_path, exist_ok=True)

        timeFlag = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        time_stamp = '_'.join(timeFlag.split())
        file_name = file_name + "_" + time_stamp
        full_path = os.path.join(full_path, file_name)

        if output_format.lower() == 'csv':
            full_path = full_path + "." + output_format.lower()
            print(full_path)
            pd_file.to_csv(full_path)
        else:
            full_path = full_path + ".xlsx"
            pd_file.to_excel(full_path, "sheet1",
                             index=False, engine='xlsxwriter')

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
        line.render('./output/render.html')


import sys
import getopt


def usage():
    print('-i: Input file, which can only be a database file')
    print('-d: The day number, can only be 1 to 3')
    print('-k: t for key actions analysis, f for all actions analysis')
    pass


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], 'hi:d:k:')
    for op, value in opts:
        if op == '-i':
            sql_in = value
        if op == '-d':
            day = int(value)
        if op == '-k':
            if value == 't':
                k = True
            else:
                k = False
        if op == '-h':
            print('help')
            usage()
    main = Main(day=day, sql_in=sql_in, k=k)
    main.ops_analysis()
