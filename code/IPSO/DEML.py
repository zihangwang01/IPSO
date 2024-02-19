import pdb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import  catboost as cb
import warnings
import heapq

from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
import copy
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
import math
from sklearn import tree
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import  accuracy_score
plt.rcParams["font.sans-serif"]='SimHei'
mpl.rcParams['axes.unicode_minus']=False
import random


class HiddenLayer:
    def __init__(self, x, num):
        row = x.shape[0]
        columns = x.shape[1]
        rnd = np.random.RandomState(4444)
        self.w = rnd.uniform(-1, 1, (columns, num))
        self.b = np.zeros([row, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            for j in range(row):
                self.b[j, i] = rand_b
        self.h = self.sigmoid(np.dot(x, self.w) + self.b)
        self.H_ = np.linalg.pinv(self.h)
        # print(self.H_.shape)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def regressor_train(self, T):
        C = 2
        I = len(T)
        sub_former = np.dot(np.transpose(self.h), self.h) + I / C
        all_m = np.dot(np.linalg.pinv(sub_former), np.transpose(self.h))
        T = T.reshape(-1, 1)
        self.beta = np.dot(all_m, T)
        return self.beta

    def classifisor_train(self, T):
        en_one = OneHotEncoder()
        T = en_one.fit_transform(T.reshape(-1, 1)).toarray()  # 独热编码之后一定要用toarray()转换成正常的数组
        C = 3
        I = len(T)
        sub_former = np.dot(np.transpose(self.h), self.h) + I / C
        all_m = np.dot(np.linalg.pinv(sub_former), np.transpose(self.h))
        self.beta = np.dot(all_m, T)
        return self.beta

    def regressor_test(self, test_x):
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        return result

    def classifisor_test(self, test_x):
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        result = [item.tolist().index(max(item.tolist())) for item in result]
        return result
x_train=pd.read_csv(r'd:\数据\x1_train_data.csv',encoding='gbk').values
x_test=pd.read_csv(r'd:\数据\x1_test_data.csv',encoding='gbk').values
y_train=pd.read_csv(r'd:\数据\y1_train_data.csv',encoding='gbk').values
y_test=pd.read_csv(r'd:\数据\y1_test_data.csv',encoding='gbk').values
a = HiddenLayer(x_train, 20)
a.classifisor_train(y_train)
list1 = a.classifisor_test(x_test)
accuracy = accuracy_score(list1, y_test)
FP = confusion_matrix(y_test, list1)[0][1] / (
            confusion_matrix(y_test, list1)[0][0] + confusion_matrix(y_test, list1)[0][1] +
            confusion_matrix(y_test, list1)[1][1] + confusion_matrix(y_test, list1)[1][0])
FN = confusion_matrix(y_test, list1)[1][0] / (
            confusion_matrix(y_test, list1)[0][0] + confusion_matrix(y_test, list1)[0][1] +
            confusion_matrix(y_test, list1)[1][1] + confusion_matrix(y_test, list1)[1][0])
print(accuracy,FP,FN)