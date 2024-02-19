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
    return accuracy, FP, FN  # Accuracy : 0.984

    # ks=KFold(n_splits=2,shuffle=True,random_state=1)# 进行2折交叉验证
        # f = 0
        # for train_index, test_index in ks.split(X):
        #
        #     decison = tree.DecisionTreeClassifier(random_state=1)  # 运用决策树来训练样本
        #     decison.fit(X[train_index], Y[train_index])
        #     y_predict = decison.predict(X[test_index])
        #     f += accuracy_score(Y[test_index], y_predict)  # 计算样本的F1 score
        # return float(f / 2)

def PSfitness_func(X,wd,x_train, x_test, y_train, y_test,t):  #sklearn拟合样本，并计算对应的准确率
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
    accuracy, FP, FN = PSfit_func(x_train1, x_test1, y_train, y_test)
    w = accuracy  # 以特征值越少越好和准确率越高越好作为选择原则
    w1 = [w, accuracy, FP, FN]
    # pdb.set_trace()
    return w1
def PSposition_update(X, V):
    """
    根据公式更新粒子的位置
    :param X: 粒子当前的位置矩阵，维度是 20*65
    :param V: 粒子当前的速度举着，维度是 20*65
    """
    return X+V
def PSbj(X,wd,size,x_train, x_test, y_train, y_test,gbest,t): #比较粒子群X中元素的适应值，选出适应值最大的位置,直接返回即可
    c = 0
    for i in range(0,size):
        # pdb.set_trace()
        m=PSfitness_func(X.iloc[i].values,wd,x_train, x_test, y_train, y_test,t)[0]
        n=PSfitness_func(X.iloc[c],wd,x_train, x_test, y_train, y_test,t)[0]
        if m >= n:
            c = i
    return c
def PSpick(X,wd,size,gbest,t,x_train, x_test, y_train, y_test):
    a=[0,0,0,0]#选取四个种群最优位置
    if t==1:
        for i in range(0, 4):
            a[i] = PSbj(X, wd, size - i,x_train, x_test, y_train, y_test,gbest,t)  # 选取适应值最大的项
            e = copy.deepcopy(X.iloc[a[i]])

            e['fit'] = copy.deepcopy(PSfitness_func(X.iloc[a[i]], wd, x_train, x_test, y_train, y_test,t)[0] ) # 将第X的第i行插入数组中
            # pdb.set_trace()
            gbest.loc[i] = copy.deepcopy(e)#为gbest的最后一列插入适应值
            # for k in range(0, wd):
            #     c = X[[a[i]],[k]]
            #     gbest[[i] ,[ k]] = c
            # pdb.set_trace()
            X=X.drop(a[i]) #在X中删去选中行
            X = X.reset_index(drop=True)  #X重置行索引
    #对gbest进行排序
        # pdb.set_trace()
        gbest=gbest.sort_values('fit',ascending=False) #对gbest按照fit值进行排序
        gbest = gbest.reset_index(drop=True)
        return gbest


    else:
        for i in range(0, 4):
            a[i] = PSbj(X, wd, size - i, x_train, x_test, y_train, y_test, gbest,t)  # 选取适应值最大的项
            e = copy.deepcopy(X.iloc[a[i]])
            e['fit']=PSfitness_func(X.iloc[a[i]], wd, x_train, x_test, y_train, y_test,t)[0]# 将第X的第i行插入数组中
            gbest.loc[i+4] = copy.deepcopy(e)  # 为gbest的最后一列插入适应值
            # for k in range(0, wd):
            #     c = X[[a[i]],[k]]
            #     gbest[[i] ,[ k]] = c
            # pdb.set_trace()
            X=X.drop(a[i], axis=0)  # 在X中删去选中行
            X = X.reset_index(drop=True) #X重置行索引
            # 对gbest进行排序
        gbest = gbest.sort_values(by=['fit'],  ascending=False)# 对gbest按照fit值进行排序
        gbest=gbest.drop_duplicates(keep='first')#删除gbest中的重复行,两条重复数据保留第一条，但行数会减少
        gbest=gbest.reset_index(drop=True)
        return gbest

def PSvelocity_update(X,pbest, gbest,t,size,max_val,wd,V,x_train, x_test, y_train, y_test):
    """
    根据速度更新公式更新每个粒子的速度
    X为粒子群当前位置
    pbest为单个粒子最优位置
    gbest为种群最优的三个位置
    size为种群中粒子的个数
    max_val为粒子最大运行速度
    wd为粒子的维度
    t为迭代次数
    """
    c1 = 1  # c1,c2为粒子对种群最优位置和个体最优位置的学习率
    c2 = 1
    w = 0.05  # w为最优位置的引导权重
    r1 = 0.03
    r2 = 0.15
    X1 = []

    for i in range(0, size):
        M = pbest.iloc[i].values.copy()  # m,n均为行向量
        # pdb.set_trace()
        N = X.iloc[i].values.copy()
        Z = gbest.iloc[0].values.copy()
        for k in range(0, wd):
            if M[k] < 0.5:
                M[k] = 0
            else:
                M[k] = 1
        for k in range(0, wd):
            if N[k] < 0.5:
                N[k] = 0
            else:
                N[k] = 1
        for k in range(0, wd):
            if Z[k] < 0.5:
                Z[k] = 0
            else:
                Z[k] = 1
        V.loc[i] = copy.deepcopy(c1 * r1 * (M - N) + c2 * r2 * (Z[0:wd] - N) + w * V.iloc[i])
    # 防止越界处理
    # V[V < -max_val] = -max_val
    # V[V > max_val] = max_val
    return V
def PSpso(x_train, x_test, y_train, y_test,fit_list,list1):
    wd = x_train.shape[1]
    size = 5  # 种群内粒子数目
    # 特征维度，并且减去因变量y
    iter_num = 50  # 最大迭代次数
    max_val = 1.0
    gbestL=np.zeros(shape=(iter_num,4,wd+1))  #保留数据点的选择情况
    X_max = 1.8
    # fitness_val_list = []  #记录最优迭代记录
    # 初始化种群各个粒子的位置(Y>0.5表示该位置的点被使用，Y<0.5表示该位置的点未被使用)
    X = np.random.rand(size, wd)  # 在（0,1）区间内生成维度为（size，wd）的浮点数
    X = pd.DataFrame(data=X, columns=list1)  # 将数据存入dataframe中
    # 初始化各个粒子的速度
    V = np.random.rand(size, wd)
    V = pd.DataFrame(data=V, columns=list1)  # 将v存入dataframe中
    # print(X)
    gbest = np.zeros(shape=(8, wd + 1))
    gbest = pd.DataFrame(data=gbest, columns=list1 + ('fit',))  # 生成gbest，最后一列为效应度函数值
    pbest = copy.deepcopy(X)  # 初始化的个体最优位置
    t = 0
    gbest = copy.deepcopy(PSpick(pbest, wd, size, gbest, t, x_train, x_test, y_train, y_test))
    # 迭代计算
    for diedai in range(0, iter_num):
        t = t + 1
        V = copy.deepcopy(PSvelocity_update(X, pbest, gbest, t, size, max_val, wd,V,x_train, x_test, y_train, y_test))
        X = copy.deepcopy(PSposition_update(X, V))
        # pdb.set_trace()
        for i in range(0, size):  # 个体现在位置与个体最优位置进行比较,并更新
            if PSfitness_func(X.iloc[i], wd,x_train, x_test, y_train, y_test,t)[0] >= PSfitness_func(pbest.iloc[i], wd,x_train, x_test, y_train, y_test,t)[0]:
                pbest.iloc[i]=copy.deepcopy(X.iloc[i])
            # 更新群体的最优位置
        gbest=copy.deepcopy(PSpick(pbest,wd,size,gbest,t,x_train, x_test, y_train, y_test))
            # 记录最优迭代记录
        geshu_1 = 0
        z=x_train.shape[1]
        for i in range(0,z):
            if gbest.iloc[0,i]>= 0.5:
                geshu_1 += 1
        geshu_2 = 0
        for i in range(0, z):
            if gbest.iloc[1,i] >= 0.5:
                geshu_2 += 1
        geshu_3 = 0

        for i in range(0, z):
            if gbest.iloc[2,i] >= 0.5:
                geshu_3 += 1
                # print("调用了第", i, "个特征")
        print("共使用了", geshu_1, "个特征,该模型的准确率为", gbest.loc[0, 'fit'], "此时Type I error为",
              PSfitness_func(gbest.iloc[0], wd, x_train, x_test, y_train, y_test, t)[2],
              "此时Type II error为", PSfitness_func(gbest.iloc[0], wd, x_train, x_test, y_train, y_test, t)[3])

        l = []
        for i in range(0,4):
            gbestL[diedai][i]=copy.deepcopy(gbest.iloc[i])
        # for i in range(1, 4 + 1):
        #     l.append(i)
        for i in range(t-1, t ):  #初始传入的t为1
            for j in range(0, size):
                fit_list[j][i] = copy.deepcopy(PSfitness_func(X.iloc[j], wd,x_train, x_test, y_train, y_test,t)[0])  # 存放历史粒子的目标值
        k=1
        # plt.scatter(l, fit_list[:, k], c='red', label=i)  # 可视化每一个粒子
        # plt.xlabel("粒子")
        # plt.ylabel("效应值")
        # plt.legend('')
        # plt.title('迭代过程')
        # plt.show()
        # k=k+1
    # 输出迭代结果
    # print("最优解是" ,gbest[0])
    # # 绘图
    # colorlist=[]
    # for i in range(0,3):
    #     j=1
    #     colorlist.append(j)
    #     j=j+1
    list = []  # 生成一个时间的自变量
    for i in range(1, iter_num + 1):  #
        list.append(int(i))
    for i in range(0,size):
        # plt.scatter(list,fit_list[i],c='r')
        cm = plt.cm.get_cmap('RdYlBu')
        sc=plt.scatter(list,fit_list[i],c=list,label=1,vmin=0,vmax=50,cmap=cm)  #为第i个粒子绘制散点图
        plt.colorbar(sc).remove()
    plt.xlabel("迭代次数")
    plt.ylabel("效应值")
    # plt.xticks([50,100,150,200,250,300,350,400,450,500])
    plt.legend('')
    plt.axhline(y=0.7, c='b', ls='--', lw='1')
    plt.title('迭代过程')
    plt.show()
    for k in range(0, iter_num):
        for z in range(4):
            for i in range(0, wd):
                if gbestL[k][z][i] > 0.5:
                    gbestL[k][z][i] = 1
                else:
                    gbestL[k][z][i] = 0
    print(gbestL[iter_num-1][0])   #输出最优粒子的特征选择情况(最后一次默认粒子群收敛)
    list1 = []  # 生成存储最优位置重要性的列表
    list2=[]
    list3=[]
    list4=[]
    for i in range(0,wd):  #
        num1 = 0
        num2=0
        num3=0
        num4=0
        for j in range(0,iter_num):
            for z in range(0,4):
                if z==0:
                    num1 += gbestL[j][z][i] * gbestL[j][z][wd]
                    list1.append(num1)
                elif z==1:
                    num2 += gbestL[j][z][i] * gbestL[j][z][wd]
                    list2.append(num2)
                elif z==2:
                    num3 += gbestL[j][z][i] * gbestL[j][z][wd]
                    list3.append(num3)
                else:
                    num4 += gbestL[j][z][i] * gbestL[j][z][wd]
                    list4.append(num4)

    l1=gbestL[iter_num-1][0]
    l2=gbestL[iter_num-1][1]
    l3 = gbestL[iter_num - 1][2]
    l4 = gbestL[iter_num - 1][3]
    dt = pd.read_csv(r'd:\test3.csv', encoding='gbk')
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    lu = dt.columns.tolist()
    shengc1 = np.array([np.arange(len(lu)), np.arange(len(lu)),np.arange(len(lu))], dtype=str).T
    shengc2 = np.array([np.arange(len(lu)), np.arange(len(lu)),np.arange(len(lu))], dtype=str).T
    shengc3 = np.array([np.arange(len(lu)), np.arange(len(lu)),np.arange(len(lu))], dtype=str).T
    shengc4 = np.array([np.arange(len(lu)), np.arange(len(lu)),np.arange(len(lu))], dtype=str).T
    for i in range(len(l1) - 1):
        shengc1[i][0] = lu[i]
        shengc1[i][1] = list1[i]
        shengc1[i][2]=l1[i]
    for i in range(len(l1) - 1):
        shengc2[i][0] = lu[i]
        shengc2[i][1] = list1[i]
        shengc2[i][2]=l1[i]
    for i in range(len(l1) - 1):
        shengc3[i][0] = lu[i]
        shengc3[i][1] = list1[i]
        shengc3[i][2]=l1[i]
    for i in range(len(l1) - 1):
        shengc4[i][0] = lu[i]
        shengc4[i][1] = list1[i]
        shengc4[i][2]=l1[i]
    dt1 = pd.DataFrame(data=shengc1, columns=("选择的特征是", "重要度为","粒子的选择情况是"))
    dt1 = dt1.sort_values(axis=0, by="重要度为", ascending=False)
    dt2 = pd.DataFrame(data=shengc2, columns=("选择的特征是", "重要度为","粒子的选择情况是"))
    dt2 = dt1.sort_values(axis=0, by="重要度为", ascending=False)
    dt3 = pd.DataFrame(data=shengc3, columns=("选择的特征是", "重要度为","粒子的选择情况是"))
    dt3 = dt1.sort_values(axis=0, by="重要度为", ascending=False)
    dt4 = pd.DataFrame(data=shengc4, columns=("选择的特征是", "重要度为","粒子的选择情况是"))
    dt4 = dt1.sort_values(axis=0, by="重要度为", ascending=False)
    dt1.to_csv(r"d:\重要性\重要性排名1.csv")
    dt2.to_csv(r"d:\重要性\重要性排名2.csv")
    dt3.to_csv(r"d:\重要性\重要性排名3.csv")
    dt4.to_csv(r"d:\重要性\重要性排名4.csv")




x_train=pd.read_csv(r'd:\数据\x1_train_data.csv',encoding='gbk')
x_test=pd.read_csv(r'd:\数据\x1_test_data.csv',encoding='gbk')
y_train=pd.read_csv(r'd:\数据\y1_train_data.csv',encoding='gbk')
y_test=pd.read_csv(r'd:\数据\y1_test_data.csv',encoding='gbk')
list=x_train.columns.values.tolist()
list1 = ()
iter_num=100
for i in list:
    list1+=(i,)
fit_list = np.zeros(shape=(20, 50))
PSpso(x_train, x_test, y_train, y_test , fit_list, list1)