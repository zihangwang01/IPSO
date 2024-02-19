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
        gbm3 = tree.DecisionTreeClassifier()
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
def jiaohuan(X,Y,k,wd):
    M=[]
    # pdb.set_trace()
    for i in range(wd):  #对于所有的基因位置
        if i <= k:
            M.append(X[i])   #则选取父亲的基因
        else:
            M.append(Y[i])
    M=np.array(M)
    return M
def mutation(child, MUTATION_RATE=0.003):
    # pdb.set_trace()
    DNA_SIZE = len(child)
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point1 = np.random.randint(0, DNA_SIZE,2)  # 随机产生两个实数，代表要变异基因的位置范围
        for i in range(min(mutate_point1),max(mutate_point1)):
            child[i]=1-child[i]
    return child
def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    # new_pop = []
    POP_SIZE=len(pop)
    DNA_SIZE=len(pop[0])
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）

        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE)  # 随机产生交叉的点
            child=jiaohuan(father,mother,cross_points,DNA_SIZE)

        child=mutation(child)  # 每个后代有一定的机率发生变异
        pop=np.insert(pop,-1,child,axis=0)    #计算适应值时需进行归一化
    # pdb.set_trace()
    return pop
def fitness(X,wd,x_train,x_test,y_train,y_test,goal):   #计算适应值大小
    fitn=[]
    goals = ['accuracy', 'Precision', 'Recall', 'F1']
    for i in range(len(X)):
        fit=PSfitness_func(X[i],wd,x_train,x_test,y_train,y_test,goals[goal])
        fitn.append(fit[goal])
    return fitn
def ga(x_train,x_test,y_train,y_test):
    pop_size=20  #一共有20条染色体
    wd=x_train.shape[1]   #染色体上有wd个基因
    maxt=50

    goals = ['accuracy', 'Precision', 'Recall', 'F1']
    for goal in range(len(goals)):
        X = np.random.rand(pop_size, wd)  # 生成染色体群体
        maxfit = 0
        MAXX = 0
        for t in tqdm(range(maxt)):
            X = np.array(crossover_and_mutation(X, CROSSOVER_RATE=0.8))  # 发生交叉互换，CROSSOVER_RATE为发生交叉互换的概率
            # pdb.set_trace()
            fitn = fitness(X, wd, x_train, x_test, y_train, y_test,goal)
            N_large = pd.DataFrame({'score': fitn}).sort_values(by='score', ascending=[False])
            id = list(N_large.index)[:20]
            # pdb.set_trace()
            maxx = X[fitn.index(max(fitn))]
            X = copy.deepcopy(X[id])  # 选取最优的20个染色体进入接下来的迭代
            maxf = max(fitn)

            if maxf > maxfit:
                maxfit = maxf
                MAXX = maxx
            fit = PSfitness_func(MAXX, wd, x_train, x_test, y_train, y_test,goals[goal])
            print("在第",t+1,"次迭代"," 对应的",goals[goal],"为",fit[goal],'犯第一类错误的概率为',
                  fit[4],'犯第二类错误的概率为',fit[5])

x_train=pd.read_csv(r'd:\new aco\Auto loan default risk\x_train.csv',encoding='gbk')
x_test=pd.read_csv(r'd:\new aco\Auto loan default risk\x_test.csv',encoding='gbk')
y_train=pd.read_csv(r'd:\new aco\Auto loan default risk\y_train.csv',encoding='gbk')
y_test=pd.read_csv(r'd:\new aco\Auto loan default risk\y_test.csv',encoding='gbk')
ga(x_train,x_test,y_train,y_test)
