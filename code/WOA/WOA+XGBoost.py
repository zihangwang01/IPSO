import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import  catboost as cb
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
import copy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
import math
from sklearn import tree
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import  accuracy_score
plt.rcParams["font.sans-serif"]='SimHei'
mpl.rcParams['axes.unicode_minus']=False
import pdb
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
    if x_train.shape[0] != 0:  # 防止没有特征被选择

        model1 = XGBClassifier(n_estimators=100, learning_rate=0.01,
                               random_state=1, eval_metric='error')
        model1.fit(x_train, y_train)
        y_pred1 = model1.predict(x_test)
        l['2'] = y_pred1
        accuracy = accuracy_score(y_pred1, y_test)
        list1 = y_pred1
        # print(a2)
        FP = confusion_matrix(y_test, list1)[0][1] / (
                confusion_matrix(y_test, list1)[0][0] + confusion_matrix(y_test, list1)[0][1] +
                confusion_matrix(y_test, list1)[1][1] + confusion_matrix(y_test, list1)[1][0])
        FN = confusion_matrix(y_test, list1)[1][0] / (
                confusion_matrix(y_test, list1)[0][0] + confusion_matrix(y_test, list1)[0][1] +
                confusion_matrix(y_test, list1)[1][1] + confusion_matrix(y_test, list1)[1][0])
        pre = precision_score(y_pred1, y_test)
        rec = recall_score(y_pred1, y_test)
        f1 = f1_score(y_pred1, y_test)
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
def danwh(X,size,wd):
    for i in range(size):
        for j in range(wd):
            if X[i][j]>0.5:
                X[i][j]=1
            else:
                X[i][j]=0     #返回特征的选择情况
    return X
def juli(i,X,size,wd):
    ju=[]
    for k in range(size):
        if k !=i:
            su=0
            for j in range(wd):
                su+=abs(X[i][j]-X[k][j])
            ju.append(su)   #把距离大小放入距离矩阵
        else:
            ju.append(1000000)
    min1=min(ju)
    return  X[ju.index(min1)]    #返回距离最近的鲸鱼位置


def xupdate(X,size,wd,t,maxt,MAXX): #更新鲸鱼位置信息
    A,C=shousuo(t,maxt)  #根据收缩包围机制生成A和C
    if abs(A)>1:#则进行搜索猎物，根据最近的鲸鱼位置进行位置的更新
        M=copy.deepcopy(danwh(X,size,wd))
        for i in range(0,size):
            L=copy.deepcopy(juli(i,M,size,wd))#计算到该鲸鱼的最近的鲸鱼位置
            D=C*L-X[i]
            X[i]=copy.deepcopy(L-A*D)

    else :  #进行螺旋更新位置(根据最优位置
        for i in range(0,size):
            D=MAXX-X[i]  #计算位置差
            l=random.random()  #生成随机数
            X[i]=copy.deepcopy(D*np.exp(l)*np.cos(2*np.pi*l)+MAXX)

    return X   #返回更新后的鲸鱼种群

def pick(X,xtrain,xtest,ytrain,ytest,size,wd,goal):   #选出最优的鲸鱼位置
    fitlist=[]
    for i in range(0,size):
        fit=PSfitness_func(X[i],wd,xtrain,xtest,ytrain,ytest,goal)  #计算每一个鲸鱼的适应值大小
        fitlist.append(fit[1])
    max1=max(fitlist)
    index1=fitlist.index(max1)  #选出最优位置
    return X[index1],PSfitness_func(X[index1],wd,xtrain,xtest,ytrain,ytest,goal)


def shousuo(t,maxt):  #根据当前迭代次数和最大迭代次数生成A
    a=2*(maxt-t)/maxt
    r=random.random() #随机生成一个数
    A=2*a*r-a
    return A,2*r

def WOA(xtrain,xtest,ytrain,ytest):
    wd=xtrain.shape[1]
    size=10
    maxt=50  #最大迭代次数
    goals = ['accuracy', 'Precision', 'Recall', 'F1']
    for goal in range(len(goals)):
        X=np.random.rand(size,wd)   #生成鲸鱼初始位置
        MAXX,MAXFIT=pick(X,xtrain,xtest,ytrain,ytest,size,wd,goals[goal])   #保存历史过程中的最优和适应值信息
        for t in tqdm(range(maxt)):
            X=copy.deepcopy(xupdate(X,size,wd,t,maxt,MAXX))
            maxx,maxfit=pick(X,xtrain,xtest,ytrain,ytest,size,wd,goals[goal])
            if maxfit[goal] >MAXFIT[goal]:
                MAXFIT=maxfit
                MAXX=maxx
            print('在第',t+1,'次迭代下',goals[goal],'为',MAXFIT[goal],'犯第一类错误的概率为',MAXFIT[4],
                  '犯第二类错误的概率为',MAXFIT[5])






x_train=pd.read_csv(r'd:\new aco\Auto loan default risk\x_train.csv',encoding='gbk')
x_test=pd.read_csv(r'd:\new aco\Auto loan default risk\x_test.csv',encoding='gbk')
y_train=pd.read_csv(r'd:\new aco\Auto loan default risk\y_train.csv',encoding='gbk')
y_test=pd.read_csv(r'd:\new aco\Auto loan default risk\y_test.csv',encoding='gbk')



WOA(x_train,x_test,y_train,y_test)

