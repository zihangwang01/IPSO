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
def PSfit_func(x_train, x_test, y_train, y_test):#调用sklearn中分类器，并计算准确率
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
    return accuracy, FP, FN,0,0,0  # Accuracy : 0.984
def PSfitness_func(X,wd,x_train, x_test, y_train, y_test):  #sklearn拟合样本，并计算对应的准确率
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
    accuracy, FP, FN,pre,rec,f1 = PSfit_func(x_train1, x_test1, y_train, y_test)
    w = accuracy  # 以特征值越少越好和准确率越高越好作为选择原则
    w1 = [w, accuracy, FP, FN,pre,rec,f1]
    # pdb.set_trace()
    return w1
def xxsupdate(xxisu,wd,size,X,fit):  #对最优方案中的信息素含量进行更新
    c=0.1
    for j in range(size):
        for i in range(wd):  # 更新所有位置的信息素浓度
            # pdb.set_trace()
            xxisu[0][i] = (1 - c) * xxisu[0][i] + fit[j] * X[j,i]
    return xxisu
def pjisuan(n,xxisu):
    a = 2  # 信息素启发因子
    b = 3  # 蚂蚁叛逆因子
    wd=len(xxisu[0])
    p=[]
    for i in range(wd):
        p.append(n[0][i]**a+xxisu[0][i]**b)
    su=sum(p)
    for i in range(wd):
        p[i]=p[i]/su
    # pdb.set_trace()
    return p

def Xshengc(n,xxisu,size,wd): #生成k个蚂蚁
    X=np.zeros(shape=(size,wd))   #生成k个未选择特征的蚂蚁
    p=pjisuan(n,xxisu)
    for i in range(size):
        # pdb.set_trace()
        idx = np.random.choice(np.arange(wd), size=wd, replace=True,
                               p=p)
        m=list(set(idx))   #生成第i个蚂蚁选择的特征
        for J in m:   #为蚂蚁选择特征
            X[i][J]+=1
    return X
def nuodate(n,wd,size,X,T):  #对最优方案中的特征选择情况更新,若所有最优方案均选择该特征，则其特征的n值越大为1,T为当前时期
    #x是最优方案的集合
    for j in range(size):
        for i in range(wd):
            n[0][i] = (X[j,i] + (T) * n[0][i]) / (T+1)

    return n


def fitness(X,wd,x_train,x_test,y_train,y_test):   #计算适应值大小
    fitn=[]
    for i in range(20):
        fit=PSfitness_func(X[i],wd,x_train,x_test,y_train,y_test)
        fitn.append(fit[0])
    return fitn
def ACO(x_train, x_test, y_train, y_test):
    tmax=50
    size=20
    wd=x_train.shape[1]
    xxisu=np.zeros(shape=(1,wd)) #生成信息素矩阵
    n=np.zeros(shape=(1,wd))
    MAXFIT=0
    MAXX=np.zeros(shape=(1,wd))
    X = np.random.rand(size,wd)
    for t in tqdm(range(tmax)):
        fitn=fitness(X, wd, x_train, x_test, y_train, y_test)
        n=nuodate(n,wd,size,X,t)
        xxisu=xxsupdate(xxisu,wd,size,X,fitn)
        maxf=max(fitn)
        maxx=X[fitn.index(maxf)]
        if maxf>MAXFIT:
            MAXFIT=maxf
            MAXX=maxx
        fit=PSfitness_func(MAXX, wd, x_train, x_test, y_train, y_test)
        print('accuracy',fit[0],'犯第一类错误的概率为',fit[2],'犯第二类错误的概率为',fit[3])
        X=Xshengc(n,xxisu,size,wd)



x_train=pd.read_csv(r'd:\数据\x1_train_data.csv',encoding='gbk')
x_test=pd.read_csv(r'd:\数据\x1_test_data.csv',encoding='gbk')
y_train=pd.read_csv(r'd:\数据\y1_train_data.csv',encoding='gbk')
y_test=pd.read_csv(r'd:\数据\y1_test_data.csv',encoding='gbk')
ACO(x_train, x_test, y_train, y_test)