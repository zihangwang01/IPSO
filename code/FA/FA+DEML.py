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
from DEML import HiddenLayer
def PSfit_func(x_train, x_test, y_train, y_test,goal):#调用sklearn中分类器，并计算准确率
    if len(x_train) !=0:
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
        return accuracy, FP, FN, 0, 0, 0  # Accuracy : 0.984
    else:
        return 0,0,0,0,0,0
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
def jul(X,Y,wd):
    d=0
    for j in range(wd):
        if X[j]>=0.5:
            X[j]=1
        elif X[j]<0.5:
            X[j]=0
        if Y[j]>=0.5:
            Y[j]=1
        elif Y[j]<0.5:
            Y[j]=0
        d+=(X[j]-Y[j])**2
    return d
def xiyinl(b,c,d):  #吸引力函数,返回吸引力大小z
    z=b*np.exp(-c*d)
    return z
def xiyin(a,b,c,X,Y,wd): #a是步长因子，b是最大吸收度,c是光吸收系数,X是被吸引的萤火虫，Y是吸引的萤火虫
    SUIJI=random.random()-0.5
    d=jul(X,Y,wd)  #计算两只萤火虫的差异性
    z=xiyinl(b,c,d)
    X=X+z*(Y-X)+SUIJI*a
    return X


def FA(xtrain,xtest,ytrain,ytest):
    maxt=100
    size=20
    b=0.04  #r=0的最大吸收度
    c=0.000001  #光吸收系数
    a=0.97  #步长因子
    wd=xtrain.shape[1]

    goals = ['accuracy']
    for goal in range(len(goals)):
        X = np.random.rand(size, wd)
        best = np.zeros(shape=(maxt, wd))
        fit = []
        for t in tqdm(range(maxt)):
            maxfit = PSfitness_func(X[0], wd, xtrain, xtest, ytrain, ytest,goals[goal])
            index = 0
            for i in range(size):
                fit = PSfitness_func(X[i], wd, xtrain, xtest, ytrain, ytest,goals[goal])  # 计算第一只萤火虫的适应值大小
                for j in range(i, size):
                    fit1 = PSfitness_func(X[j], wd, xtrain, xtest, ytrain, ytest,goals[goal])  # 计算另一只萤火虫的适应值大小
                    if fit1[goal] > fit[goal]:  # 若第一只萤火虫的亮度小于另一只萤火虫，则该萤火虫被吸引
                        X[i] = xiyin(a, b, c, X[i], X[j], wd)
                        fit = PSfitness_func(X[i], wd, xtrain, xtest, ytrain, ytest,goals[goal])  # 更新粒子适应度
                if maxfit[goal] < fit[goal]:  # 更新最大适应值和对应的索引
                    maxfit = fit
                    index = i
            print(t + 1)
            print('在第',t,'次迭代下',goals[goal],'为',maxfit[goal],'犯第一类错误的概率为',maxfit[4],'犯第二类错误的概率为',maxfit[5])
            best[t] = copy.deepcopy(X[index])
            fit.append(maxfit[goal])
    zyx = []
    for i in range(wd):  # 存储粒子重要性信息
        su = 0
        for j in range(maxt):
            su += best[j][i] * fit[j]
        zyx.append(su)
    dt = pd.read_csv(r'd:\test3.csv', encoding='gbk')
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    lu = dt.columns.tolist()
    shengc1 = np.array([np.arange(len(lu)), np.arange(len(lu)), np.arange(len(lu))], dtype=str).T
    for i in range(wd):
        shengc1[i][0] = lu[i]
        shengc1[i][1] = zyx[i]
        if best[maxt-1][i]>0.5:
            best[maxt-1][i]=1
        else:
            best[maxt-1][i]=0
        shengc1[i][2] = best[maxt - 1][i]  # 最后粒子的选择情况
    dt1 = pd.DataFrame(data=shengc1, columns=("选择的特征是", "重要度为", "粒子的选择情况是"))
    dt1 = dt1.sort_values(axis=0, by="重要度为", ascending=False)
    dt1.to_csv(r"重要性排名1.csv")

x_train=pd.read_csv(r'd:\数据\x1_train_data.csv',encoding='gbk')
x_test=pd.read_csv(r'd:\数据\x1_test_data.csv',encoding='gbk')
y_train=pd.read_csv(r'd:\数据\y1_train_data.csv',encoding='gbk')
y_test=pd.read_csv(r'd:\数据\y1_test_data.csv',encoding='gbk')
FA(x_train,x_test,y_train,y_test)
