import xgboost as xgb
import sklearn
from sklearn import metrics
import numpy as np
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest


class XGB_Model():
    '''
    '''

    def __init__(self, X, Y, op_name, validate_size, test_size):
        self.X = X
        self.Y = Y
        self.op_name = op_name
        self.key_ops = None
        X_train, self.X_test, Y_train, self.Y_test = train_test_split(
            X, Y, test_size=test_size)
        self.X_train, self.X_validate, self.Y_train, self.Y_validate = train_test_split(
            X_train, Y_train, test_size=validate_size)

    def model(self):                
        model = xgb.XGBClassifier(
            learning_rate=0.1, n_estimators=20, max_depth=3, subsample=1)
        eval_set = [(self.X_validate, self.Y_validate)]
        model.fit(self.X_train, self.Y_train, early_stopping_rounds=20,
                  eval_metric="logloss", eval_set=eval_set, verbose=True)

        # Y_pred -> np.ndarray Y_train -> list
        Y_pred = model.predict(self.X_train)
        print('training score {}'.format(accuracy_score(self.Y_train, Y_pred)))
        Y_pred = model.predict(self.X_validate)
        print('validate score {}'.format(
            accuracy_score(self.Y_validate, Y_pred)))
        Y_pred = model.predict(self.X_test)
        print('test score {}'.format(accuracy_score(self.Y_test, Y_pred)))
        print(precision_recall_fscore_support(
            self.Y_test, Y_pred, average=None))

        print(np.shape(model.feature_importances_))
        self.key_ops = list(model.feature_importances_)
