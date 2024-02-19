import pdb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import  catboost as cb
import warnings
import heapq
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
def PSfit_func(x_train, x_test, y_train, y_test,goal):#调用sklearn中分类器，并计算准确率
    l = pd.DataFrame(columns=('1', '2', '3', '4', '5'))
    if x_train.shape[0] >= 100:  # 防止没有特征被选择
        gbm3 = GradientBoostingClassifier(learning_rate=0.123, n_estimators=41,
                                          min_samples_leaf=60,
                                          subsample=0.7, random_state=1)
        gbm3.fit(x_train, y_train)
        y_pred = gbm3.predict(x_test)
        l['1'] = y_pred
        accuracy = accuracy_score(y_test, y_pred)
        # print(a1)
        FP = confusion_matrix(y_test, y_pred)[0][1] / (
                confusion_matrix(y_test, y_pred)[0][0] + confusion_matrix(y_test, y_pred)[0][1] +
                confusion_matrix(y_test, y_pred)[1][1] + confusion_matrix(y_test, y_pred)[1][0])
        FN = confusion_matrix(y_test, y_pred)[1][0] / (
                confusion_matrix(y_test, y_pred)[0][0] + confusion_matrix(y_test, y_pred)[0][1] +
                confusion_matrix(y_test, y_pred)[1][1] + confusion_matrix(y_test, y_pred)[1][0])
        pre = precision_score(y_pred, y_test)
        rec = recall_score(y_pred, y_test)
        f1 = f1_score(y_pred, y_test)
        return accuracy, FP, FN, pre, rec, f1  # Accuracy : 0.984
    else:
        return 0, 0, 0, 0, 0, 0
def PSfitness_func(X,wd,x_train, x_test, y_train, y_test,goal):  #sklearn拟合样本，并计算对应的准确率
    """计算单个粒子的的适应度值，也就是目标函数值 """
    # train_inf = np.isinf(X1)
    # X1[train_inf] = 0
    # n=X1.shape[0]
    # name=X1.columns
    # name=name[0:n:1]
    # X=X.values
    columns = x_train.columns.values.tolist()  # 获取列名
    X_train = np.array(x_train)
    X_test = np.array(x_test)
    y_test = np.array(y_test)
    y_train = np.array(y_train)  # 选取因变量

    co = ()
    for i in range(0, len(columns) - 1):
        co += (columns[i],)  # 将列名放到元组中
    x_train1 = pd.DataFrame(columns=co)  # 构建索引为columns的二维数组
    x_test1 = pd.DataFrame(columns=co)
    for i in range(0, wd):
        if X[i] >= 0.5:
            # pdb.set_trace()
            x_train1[columns[i]] = copy.deepcopy(X_train[:, i])  # 若第i个特征值被选取，则增加该列
    for i in range(0, wd):
        if X[i] >= 0.5:
            x_test1[columns[i]] = copy.deepcopy(X_test[:, i])  # 若第i个特征值被选取，则增加该列
    m = 0  # 设置空的二维数组，用来存储调用的数据
    x_train1 = x_train1.dropna(axis=1)  # 删去含空值的行
    x_train1 = np.array(x_train1)
    x_test1 = x_test1.dropna(axis=1)  # 删去含空值的行
    x_test1 = np.array(x_test1)
    geshu = x_train1.shape[1]  # 查看调用了多少个X
    accuracy, FP, FN,pre,rec,f1 = PSfit_func(x_train1, x_test1, y_train, y_test,goal)
    w = accuracy  # 以特征值越少越好和准确率越高越好作为选择原则
    w1 = [ accuracy,pre,rec,f1, FP, FN]
    # pdb.set_trace()
    return w1
def jiaohuan(x,wd):  #对随机的一个位置进行交叉互换
    z=np.random.randint(0,wd-1)
    x[z]=1-x[z]
    return x
def monituihuo(xtrain,xtest,ytrain,ytest):
    goals = ['accuracy', 'Precision', 'Recall', 'F1']
    for goal in range(len(goals)):
        T0 = 10
        t_min = 0.0001
        wd = xtrain.shape[1]
        X = np.random.randint(2, size=(1, wd))  # 生成初始位置
        x = X[0]

        MAXX = copy.deepcopy(X[0])
        fit1 = PSfitness_func(x, wd, xtrain, xtest, ytrain, ytest,goals[goal])
        MAXFIT=fit1
        i = 1
        while (T0 > t_min):

            x1 = jiaohuan(x, wd)
            fit2 = PSfitness_func(x1, wd, xtrain, xtest, ytrain, ytest,goals[goal])
            d = fit2[goal] - fit1[goal]
            if fit2[goal] > fit1[goal]:  # 若适应值变大则采用该位置
                x = x1
            else:
                if np.exp(-d / T0) > np.random.rand():  # 如果该概率大于随机概率则也认为存在位置更替
                    x = x1
            if MAXFIT[goal] <fit2[goal]:
                MAXX=copy.deepcopy(x1)
                MAXFIT=fit2
            fit1 = PSfitness_func(MAXX, wd, xtrain, xtest, ytrain, ytest,goals[goal])
            T0 = T0 / (i)
            i += 1

            print('在第',i-1,'次迭代时',goals[goal],'为',fit1[goal],'犯第一类错误的概率为',fit1[4],'犯第二类错误的概率为',fit1[5])


x_train=pd.read_csv(r'd:\new aco\Auto loan default risk\x_train.csv',encoding='gb18030')
x_test=pd.read_csv(r'd:\new aco\Auto loan default risk\x_test.csv',encoding='gb18030')
y_train=pd.read_csv(r'd:\new aco\Auto loan default risk\y_train.csv',encoding='gb18030')
y_test=pd.read_csv(r'd:\new aco\Auto loan default risk\y_test.csv',encoding='gb18030')
monituihuo(x_train, x_test,y_train,y_test)


