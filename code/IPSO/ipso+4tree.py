import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import  catboost as cb
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
import copy
from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
import math
# from sklearn import tree
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import  accuracy_score
plt.rcParams["font.sans-serif"]='SimHei'
mpl.rcParams['axes.unicode_minus']=False
import pdb
from tqdm import tqdm
def I4fit_func(x_train, x_test, y_train, y_test):  # 调用sklearn中分类器，并计算准确率
    l = pd.DataFrame(columns=('1', '2', '3', '4', '5'))
    if x_train.shape[0] != 0:  # 防止没有特征被选择
        for i in range(0, 4):
            if i == 0:  # 调用GDBT
                gbm3 = GradientBoostingClassifier(learning_rate=0.053, n_estimators=63,
                                                  random_state=1)
                gbm3.fit(x_train, y_train)
                y_pred = gbm3.predict(x_test)
                l['1'] = y_pred
                a1 = accuracy_score(y_test, y_pred)
                # print(a1)
            elif i == 1:  # 调用catboost
                model1 = cb.CatBoostClassifier(iterations=103, verbose=False,max_depth=7,
                                               learning_rate=0.033, random_state=1)
                model1.fit(x_train, y_train)
                y_pred1 = model1.predict(x_test)
                l['2'] = y_pred1
                a2 = accuracy_score(y_pred1, y_test)
                # print(a2)

            elif i == 2:  # 调用XGBclassifier
                model2 = XGBClassifier(n_estimators=63, learning_rate=0.053,
                                       random_state=1, eval_metric='error')
                model2.fit(x_train, y_train)
                y_pred2 = model2.predict(x_test)
                l['3'] = y_pred2
                a3 = accuracy_score(y_pred2, y_test)
                # print(a3)
            elif i == 3:
                model3 = LGBMClassifier(n_estimators=63,learning_rate=0.053,
                                        random_state=1)
                model3.fit(x_train, y_train)
                y_pred3 = model3.predict(x_test)
                l['4'] = y_pred3
                a4 = accuracy_score(y_pred3, y_test)
                # print(a4)
        l = l.replace(0, -1)  # 将l中的0全换成-1
        list1 = []
        m = [a1, a2, a3, a4]
        m = sorted(m)  # 按大小进行排序
        dis = m[3] - m[0]
        for i in range(0, l.shape[0]):
            if 0.015 > dis > 0.005:
                if l.loc[i, '1'] * l.loc[i, '2'] * l.loc[i, '3'] * l.loc[i, '4'] >= 0:
                    if a1 * l.loc[i, '1'] + a2 * l.loc[i, '2'] + a3 * l.loc[i, '3'] + a4 * l.loc[i, '4'] >= 0:
                        list1.append(1)
                    else:
                        list1.append(0)
                else:
                    if l.loc[i, '1'] < 0 and l.loc[i, '2'] == l.loc[i, '3'] == l.loc[
                        i, '4'] == 1 and a1 > a2 and a1 > a3 and a1 > a4:
                        list1.append(0)
                    elif l.loc[i, '1'] > 0 and l.loc[i, '2'] == l.loc[i, '3'] == l.loc[
                        i, '4'] == -1 and a1 > a2 and a1 > a3 and a1 > a4:
                        list1.append(1)
                    elif l.loc[i, '2'] < 0 and l.loc[i, '1'] == l.loc[i, '3'] == l.loc[
                        i, '4'] == 1 and a2 > a1 and a2 > a3 and a2 > a4:
                        list1.append(0)
                    elif l.loc[i, '2'] > 0 and l.loc[i, '1'] == l.loc[i, '3'] == l.loc[
                        i, '4'] == -1 and a2 > a1 and a2 > a3 and a2 > a4:
                        list1.append(1)
                    elif l.loc[i, '3'] == -1 and l.loc[i, '1'] == l.loc[i, '2'] == l.loc[
                        i, '4'] == 1 and a3 > a1 and a3 > a2 and a3 > a4:
                        list1.append(0)
                    elif l.loc[i, '3'] > 0 and l.loc[i, '1'] == l.loc[i, '2'] == l.loc[
                        i, '4'] == -1 and a3 > a1 and a3 > a2 and a3 > a4:
                        list1.append(1)
                    elif l.loc[i, '4'] == -1 and l.loc[i, '1'] == l.loc[i, '2'] == l.loc[
                        i, '3'] == 1 and a4 > a1 and a4 > a2 and a4 > a3:
                        list1.append(0)
                    elif l.loc[i, '4'] > 0 and l.loc[i, '1'] == l.loc[i, '2'] == l.loc[
                        i, '3'] == -1 and a4 > a1 and a4 > a2 and a4 > a3:
                        list1.append(1)
                    else:
                        if a1 * l.loc[i, '1'] + a2 * l.loc[i, '2'] + a3 * l.loc[i, '3'] + a4 * l.loc[i, '4'] >= 0:
                            list1.append(1)
                        else:
                            list1.append(0)
            elif dis > 0.015:
                if a1 > a2 and a1 > a3 and a1 > a4:
                    A1 = 1.5 * a1
                    A2 = a2
                    A3 = a3
                    A4 = a4
                elif a2 > a1 and a2 > a3 and a2 > a4:
                    A2 = 1.5 * a2
                    A1 = a1
                    A3 = a3
                    A4 = a4
                elif a3 > a2 and a3 > a1 and a3 > a4:
                    A3 = 1.5 * a3
                    A2 = a2
                    A1 = a1
                    A4 = a4
                elif a4 > a2 and a4 > a3 and a4 > a1:
                    A4 = 1.5 * a4
                    A2 = a2
                    A3 = a3
                    A1 = a1
                else:
                    A1 = a1
                    A2 = a2
                    A3 = a3
                    A4 = a4
                if l.loc[i, '1'] * l.loc[i, '2'] * l.loc[i, '3'] * l.loc[i, '4'] >= 0:
                    if A1 * l.loc[i, '1'] + A2 * l.loc[i, '2'] + A3 * l.loc[i, '3'] + A4 * l.loc[i, '4'] >= 0:
                        list1.append(1)
                    else:
                        list1.append(0)
                else:
                    if l.loc[i, '1'] < 0 and l.loc[i, '2'] == l.loc[i, '3'] == l.loc[
                        i, '4'] == 1 and a1 > a2 and a1 > a3 and a1 > a4:
                        list1.append(0)
                    elif l.loc[i, '1'] > 0 and l.loc[i, '2'] == l.loc[i, '3'] == l.loc[
                        i, '4'] == -1 and a1 > a2 and a1 > a3 and a1 > a4:
                        list1.append(1)
                    elif l.loc[i, '2'] < 0 and l.loc[i, '1'] == l.loc[i, '3'] == l.loc[
                        i, '4'] == 1 and a2 > a1 and a2 > a3 and a2 > a4:
                        list1.append(0)
                    elif l.loc[i, '2'] > 0 and l.loc[i, '1'] == l.loc[i, '3'] == l.loc[
                        i, '4'] == -1 and a2 > a1 and a2 > a3 and a2 > a4:
                        list1.append(1)
                    elif l.loc[i, '3'] == -1 and l.loc[i, '1'] == l.loc[i, '2'] == l.loc[
                        i, '4'] == 1 and a3 > a1 and a3 > a2 and a3 > a4:
                        list1.append(0)
                    elif l.loc[i, '3'] > 0 and l.loc[i, '1'] == l.loc[i, '2'] == l.loc[
                        i, '4'] == -1 and a3 > a1 and a3 > a2 and a3 > a4:
                        list1.append(1)
                    elif l.loc[i, '4'] == -1 and l.loc[i, '1'] == l.loc[i, '2'] == l.loc[
                        i, '3'] == 1 and a4 > a1 and a4 > a2 and a4 > a3:
                        list1.append(0)
                    elif l.loc[i, '4'] > 0 and l.loc[i, '1'] == l.loc[i, '2'] == l.loc[
                        i, '3'] == -1 and a4 > a1 and a4 > a2 and a4 > a3:
                        list1.append(1)
                    else:
                        if a1 * l.loc[i, '1'] + a2 * l.loc[i, '2'] + a3 * l.loc[i, '3'] + a4 * l.loc[i, '4'] >= 0:
                            list1.append(1)
                        else:
                            list1.append(0)

            else:
                if a1 * l.loc[i, '1'] + a2 * l.loc[i, '2'] + a3 * l.loc[i, '3'] + a4 * l.loc[i, '4'] >= 0:
                    list1.append(1)
                else:
                    list1.append(0)

        accuracy = accuracy_score(list1, y_test)
        FP = confusion_matrix(y_test, list1)[0][1] / (
                confusion_matrix(y_test, list1)[0][0] + confusion_matrix(y_test, list1)[0][1] +
                confusion_matrix(y_test, list1)[1][1] + confusion_matrix(y_test, list1)[1][0])
        FN = confusion_matrix(y_test, list1)[1][0] / (
                confusion_matrix(y_test, list1)[0][0] + confusion_matrix(y_test, list1)[0][1] +
                confusion_matrix(y_test, list1)[1][1] + confusion_matrix(y_test, list1)[1][0])
        return accuracy, FP, FN  # Accuracy : 0.984
    else:
        return 0, 0, 0

    # ks=KFold(n_splits=2,shuffle=True,random_state=1)# 进行2折交叉验证
    # f = 0
    # for train_index, test_index in ks.split(X):
    #
    #     decison = tree.DecisionTreeClassifier(random_state=1)  # 运用决策树来训练样本
    #     decison.fit(X[train_index], Y[train_index])
    #     y_predict = decison.predict(X[test_index])
    #     f += accuracy_score(Y[test_index], y_predict)  # 计算样本的F1 score
    # return float(f / 2)


def I4fitness_func(X, wd, x_train, x_test, y_train, y_test , t):  # sklearn拟合样本，并计算对应的准确率
    """计算单个粒子的的适应度值，也就是目标函数值 """
    # train_inf = np.isinf(X1)
    # X1[train_inf] = 0
    # n=X1.shape[0]
    # name=X1.columns
    # name=name[0:n:1]
    # X=X.values
    columns = x_train.columns.values.tolist()  # 获取列名
    X_train = np.array(x_train)
    X_test=np.array(x_test)
    y_test=np.array(y_test)
    y_train=np.array(y_train)# 选取因变量

    co = ()
    for i in range(0, len(columns) - 1):
        co += (columns[i],)  # 将列名放到元组中
    x_train1 = pd.DataFrame(columns=co)  # 构建索引为columns的二维数组
    x_test1=pd.DataFrame(columns=co)
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
    accuracy, FP, FN = I4fit_func(x_train1, x_test1, y_train, y_test)
    w = accuracy  # 以特征值越少越好和准确率越高越好作为选择原则
    w1 = [w, accuracy, FP, FN]
    # pdb.set_trace()
    return w1


def I4position_update(X, V):
    """
    根据公式更新粒子的位置
    :param X: 粒子当前的位置矩阵，维度是 20*65
    :param V: 粒子当前的速度举着，维度是 20*65
    """
    return X + V


# 计算粒子间的距离
def I4ed(m, n, wd):  # m为单个粒子，n为种群最优粒子
    x = 0  # 当粒子在第i个位置取值大于0.5则定义为1；反之定义为0，从而计算距离
    M = m.values.copy()  # m,n均为行向量
    # pdb.set_trace()
    N = n.values.copy()
    for i in range(0, wd):
        if M[i] < 0.5:
            M[i] = 0
        else:
            M[i] = 1
    for i in range(0, wd):
        if N[i] < 0.5:
            N[i] = 0
        else:
            N[i] = 1
    for i in range(0, wd):
        x += (M[i] - N[i]) ** 2

    return x


# 最差粒子更新公式
def I4worst_update(X, size, wd, x_train, x_test, y_train, y_test , list1, t):
    a = 0
    b = [0, 0, 0]
    X1 = np.zeros(shape=(4, wd))
    X1 = pd.DataFrame(X1, columns=list1)  # 存放最劣粒子
    # 选出最劣点
    for i in range(0, size):
        if I4fitness_func(X.iloc[i], wd, x_train, x_test, y_train, y_test , t)[0] <= I4fitness_func(X.iloc[a], wd, x_train, x_test, y_train, y_test , t)[0]:
            a = i
    X1.iloc[0] = copy.deepcopy(X.iloc[a])
    X = X.drop(a)
    X = X.reset_index(drop=True)
    # 选出离最劣点最近的三个点
    for j in range(0, size - 1):
        if I4ed(X1.iloc[0], X.iloc[j], wd) <= I4ed(X1.iloc[0], X.iloc[b[0]], wd):
            b[0] = j
    X1.iloc[1] = copy.deepcopy(X.iloc[b[0]])
    X = X.drop(b[0])
    X = X.reset_index(drop=True)
    for j in range(0, size - 2):
        if I4ed(X1.iloc[0], X.iloc[j], wd) <= I4ed(X1.iloc[0], X.iloc[b[1]], wd):
            b[1] = j
    X1.iloc[2] = copy.deepcopy(X.iloc[b[1]])
    X = X.drop(b[1])
    X = X.reset_index(drop=True)
    for j in range(0, size - 2 - 1):
        if I4ed(X1.iloc[0], X.iloc[j], wd) <= I4ed(X1.iloc[0], X.iloc[b[2]], wd):
            b[2] = j
    X1.iloc[3] = copy.deepcopy(X.iloc[b[2]])
    X = X.drop(b[2])
    X = X.reset_index(drop=True)

    # 此时最差点是X1[a]，与该点距离最近的三个点为X1[b[0]],X1[b[1]]，X2[b[2]]
    X2 = copy.deepcopy(X1)  # 复制一份列表，X2中放改良得到的四个粒子
    for j in range(0, 3):
        m = copy.deepcopy(X2.iloc[0])  # 提取最差点
        k = np.random.randint(0, wd, 1)
        z = copy.deepcopy(X1.iloc[j + 1, k])
        # pdb.set_trace()
        m[k] = z  # 最差点学习亲朋行为
        X2.iloc[j + 1] = copy.deepcopy(m)
    c = 0
    for i in range(0, 4):  # 选出更新后最好点
        if I4fitness_func(X2.iloc[i], wd, x_train, x_test, y_train, y_test , t)[0] <= I4fitness_func(X2.iloc[c], wd, x_train, x_test, y_train, y_test , t)[0]:
            c = i
    X1.iloc[0] = copy.deepcopy(X2.iloc[c])  # 最差点的更新
    X = pd.concat([X, X1], ignore_index=True)  # 将更新结果传入粒子群
    return X


def I4paichi(X, gbest, wd, t, size):
    # 假设半径为3，在t(0)时刻，不产生排斥力的点的最大数量为5,r=0.2
    # 在t时刻 生成半径为2的空间内点的个数的计算公式
    r = 0.0114
    xmax = math.ceil(size / (1 + (size / 3 - 1) * np.exp(-r * t)))  # 利用logist函数 生成t时刻的种群最大数量，并向上取整
    print(xmax)
    w = 0
    for i in range(0, size):  # 计算种群中粒子与最优点距离在要求半径以内的粒子个数 该步骤只考虑种群最优位置
        # pdb.set_trace()
        m = I4ed(X.iloc[i], gbest.iloc[0], wd)
        if m <= 25:
            w = w + 1
    print(w)

    # pdb.set_trace()
    if xmax != size:
        if w >= 0.8* xmax and w <= 1 * xmax:
            p = (np.exp(xmax - w)) / (1 + (np.exp(xmax - w)))
        elif w > xmax:
            p = -((w / (xmax + 1))) * (np.exp((w - xmax))) / (1 + (np.exp(w - xmax)))

        else:
            p = (xmax / (w + 1)) * (np.exp(xmax - w)) / (1 + (np.exp(xmax - w)))

        return p
    else:
        return 1.2


def I4My(dis, fit):  # 构建评价函数（与适应度正相关，与距离反相关）
    a = 0.4
    b = 0.005
    return abs(a * fit) - b * dis


def I4bj(X, wd, size, x_train, x_test, y_train, y_test , gbest, t):  # 比较粒子群X中元素的适应值，选出适应值最大的位置,直接返回即可
    c = 0
    for i in range(0, size):
        # pdb.set_trace()
        m = I4fitness_func(X.iloc[i].values, wd, x_train, x_test, y_train, y_test , t)[0]
        n = I4fitness_func(X.iloc[c], wd, x_train, x_test, y_train, y_test , t)[0]
        if m >= n:
            c = i
    return c


# def bj1(gbest,wd,size,XZQ):
#     a=0
#     for i in range(0,size):
#         m=fitness_func(gbest[i],wd,XZQ)[0]
#         n=fitness_func(gbest[a],wd,XZQ)[0]
#         if m>=fitness_func(gbest[a],wd,XZQ)[0]:  #从最优种群选出最优的4个粒子
#             a=copy.deepcopy(i)
#     return a

def I4pick(X, wd, size, gbest, t, x_train, x_test, y_train, y_test ):
    a = [0, 0, 0, 0]  # 选取四个种群最优位置
    if t == 1:
        for i in range(0, 4):
            a[i] = I4bj(X, wd, size - i, x_train, x_test, y_train, y_test , gbest, t)  # 选取适应值最大的项
            e = copy.deepcopy(X.iloc[a[i]])

            e['fit'] = copy.deepcopy(I4fitness_func(X.iloc[a[i]], wd, x_train, x_test, y_train, y_test , t)[0])  # 将第X的第i行插入数组中
            # pdb.set_trace()
            gbest.loc[i] = copy.deepcopy(e)  # 为gbest的最后一列插入适应值
            # for k in range(0, wd):
            #     c = X[[a[i]],[k]]
            #     gbest[[i] ,[ k]] = c
            # pdb.set_trace()
            X = X.drop(a[i])  # 在X中删去选中行
            X = X.reset_index(drop=True)  # X重置行索引
        # 对gbest进行排序
        # pdb.set_trace()
        gbest = gbest.sort_values('fit', ascending=False)  # 对gbest按照fit值进行排序
        gbest = gbest.reset_index(drop=True)
        return gbest


    else:
        for i in range(0, 4):
            a[i] = I4bj(X, wd, size - i, x_train, x_test, y_train, y_test , gbest, t)  # 选取适应值最大的项
            e = copy.deepcopy(X.iloc[a[i]])
            e['fit'] = I4fitness_func(X.iloc[a[i]], wd, x_train, x_test, y_train, y_test , t)[0]  # 将第X的第i行插入数组中
            gbest.loc[i + 4] = copy.deepcopy(e)  # 为gbest的最后一列插入适应值
            # for k in range(0, wd):
            #     c = X[[a[i]],[k]]
            #     gbest[[i] ,[ k]] = c
            # pdb.set_trace()
            X = X.drop(a[i], axis=0)  # 在X中删去选中行
            X = X.reset_index(drop=True)  # X重置行索引
            # 对gbest进行排序
        gbest = gbest.sort_values(by=['fit'], ascending=False)  # 对gbest按照fit值进行排序
        gbest = gbest.drop_duplicates(keep='first')  # 删除gbest中的重复行,两条重复数据保留第一条，但行数会减少
        gbest = gbest.reset_index(drop=True)
        return gbest


def I4yind(gbest, y, wd, x_train, x_test, y_train, y_test , t):  # 计算每一个粒子的引导策略，其中gbest是最优的四个粒子,size是种群个数，wd是空间维度,y是受引导的粒子
    A = [0, 0, 0]
    for i in range(0, 3):
        A[i] = I4My(I4ed(gbest.iloc[i], y, wd), I4fitness_func(gbest.iloc[i], wd, x_train, x_test, y_train, y_test , t)[0])
    m = 0
    for i in A:
        m += i
    Z = copy.deepcopy(y)
    M = copy.deepcopy(gbest)
    for k in range(0, 3):
        for i in range(0, wd):  # 清除量纲的影响
            if M.iloc[k, i] < 0.5:
                M.iloc[k, i] = 0
            else:
                M.iloc[k, i] = 1
    for i in range(0, wd):  # 清除量纲的影响
        if Z[i] < 0.5:
            Z[i] = 0
        else:
            Z[i] = 1
    Y = (A[0] * (M.iloc[0, :wd] - Z) + A[1] * (M.iloc[1, :wd] - Z) + A[2] * (M.iloc[2] - Z)) / m

    return Y


def I4velocity_update(X, pbest, gbest, t, size, max_val, wd, V, x_train, x_test, y_train, y_test ):
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
    w = 0.04  # w为最优位置的引导权重
    p = I4paichi(X, gbest, wd, t, size)
    r1 = 0.05
    r2 = 0.1  # 决定排斥力大小
    X1 = []
    Z = gbest.iloc[0].values.copy()
    for k in range(0, wd):
        if Z[k] < 0.5:
            Z[k] = 0
        else:
            Z[k] = 1
    # if p <0:
    #     pdb.set_trace()
    if t > 80:
        g = 0
    else:
        g = 0.8 - t * 0.002
    for i in range(0, size):
        M = pbest.iloc[i].values.copy()  # m,n均为行向量
        # pdb.set_trace()
        N = X.iloc[i].values.copy()

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

        # pdb.set_trace()
        V.loc[i] = copy.deepcopy(
            (g * w * I4yind(gbest, X.iloc[i], wd, x_train, x_test, y_train, y_test , t).values[1:wd + 1]) + c1 * r1 * (M - N) + c2 * r2 * p * (
                    Z[0:wd] - N) + (1 - g) * w * V.iloc[i])
    for i in range(0, size):  # 防止速度大小过大
        for j in range(0, wd):
            if -max_val <= V.iloc[i, j] <= max_val:
                V.iloc[i, j] = V.iloc[i, j]
            elif V.iloc[i, j] > max_val:
                V.iloc[i, j] = max_val
            elif V.iloc[i, j] < - max_val:
                V.iloc[i, j] = -max_val
    return V


def I4pso(x_train, x_test, y_train, y_test , fit_list, list1):
    wd = x_train.shape[1]
    size = 20  # 种群内粒子数目
    # 特征维度，并且减去因变量y
    iter_num = 500  # 最大迭代次数
    max_val = 0.7
    X_max = 1.8
    # fitness_val_list = []  #记录最优迭代记录
    # 初始化种群各个粒子的位置(Y>0.5表示该位置的点被使用，Y<0.5表示该位置的点未被使用)
    X = np.random.rand(size, wd)  # 在（0,1）区间内生成维度为（size，wd）的浮点数
    X = pd.DataFrame(data=X, columns=list1)  # 将数据存入dataframe中
    gbestL = np.zeros(shape=(iter_num, 4, wd + 1))  # 保留数据点的选择情况
    # 初始化各个粒子的速度

    V = np.random.uniform(-0.2, 0.2, (size, wd))
    V = pd.DataFrame(data=V, columns=list1)
    # print(X)
    gbest = np.zeros(shape=(8, wd + 1))
    gbest = pd.DataFrame(data=gbest, columns=list1 + ('fit',))  # 生成gbest，最后一列为效应度函数值
    pbest = copy.deepcopy(X)  # 初始化的个体最优位置
    t = 0
    gbest = copy.deepcopy(I4pick(pbest, wd, size, gbest, t, x_train, x_test, y_train, y_test ))  # 选取四个种群最优位置
    # 迭代计算
    for diedai in tqdm(range(0, iter_num)):
        t = t + 1
        X = copy.deepcopy(I4worst_update(X, size, wd, x_train, x_test, y_train, y_test , list1, t))  # 最差点更新
        V = copy.deepcopy(I4velocity_update(X, pbest, gbest, t, size, max_val, wd, V, x_train, x_test, y_train, y_test ))
        X = copy.deepcopy(I4position_update(X, V))
        for i in range(0, size):  # 防止位置大小过大
            for j in range(0, wd):
                if -X_max <= X.iloc[i, j] <= X_max:
                    X.iloc[i, j] = X.iloc[i, j]
                elif X.iloc[i, j] > X_max:
                    X.iloc[i, j] = X_max
                else:
                    X.iloc[i, j] = -X_max
        # pdb.set_trace()
        for i in range(0, size):  # 个体现在位置与个体最优位置进行比较,并更新
            if I4fitness_func(X.iloc[i], wd, x_train, x_test, y_train, y_test , t)[0] >= I4fitness_func(pbest.iloc[i], wd, x_train, x_test, y_train, y_test , t)[0]:
                pbest.iloc[i] = copy.deepcopy(X.iloc[i])
            # 更新群体的最优位置
        gbest = copy.deepcopy(I4pick(pbest, wd, size, gbest, t, x_train, x_test, y_train, y_test ))
        # 记录最优迭代记录
        geshu_1 = 0
        z = x_train.shape[1]
        for i in range(0, z):
            if gbest.iloc[0, i] >= 0.5:
                geshu_1 += 1
        geshu_2 = 0
        for i in range(0, z):
            if gbest.iloc[1, i] >= 0.5:
                geshu_2 += 1
        geshu_3 = 0

        for i in range(0, z):
            if gbest.iloc[2, i] >= 0.5:
                geshu_3 += 1
                # print("调用了第", i, "个特征")
        print("共使用了", geshu_1, "个特征,该模型的准确率为", gbest.loc[0, 'fit'], "此时Type I error为",
              I4fitness_func(gbest.iloc[0], wd, x_train, x_test, y_train, y_test , t)[2],
              "此时Type II error为", I4fitness_func(gbest.iloc[0], wd, x_train, x_test, y_train, y_test , t)[3])
        print("共使用了", geshu_2, "个特征,该模型的准确率为", gbest.loc[1, 'fit'], "此时Type I error为",
              I4fitness_func(gbest.iloc[1], wd, x_train, x_test, y_train, y_test , t)[2],
              "此时Type II error为", I4fitness_func(gbest.iloc[1], wd, x_train, x_test, y_train, y_test , t)[3])
        print("共使用了", geshu_2, "个特征,该模型的准确率为", gbest.loc[1, 'fit'], "此时Type I error为",
              I4fitness_func(gbest.iloc[2], wd, x_train, x_test, y_train, y_test , t)[2],
              "此时Type II error为", I4fitness_func(gbest.iloc[2], wd, x_train, x_test, y_train, y_test , t)[3])
        for i in range(0,4):
            gbestL[diedai][i]=copy.deepcopy(gbest.iloc[i])
        l = []
        for i in range(1, size + 1):
            l.append(i)
        for i in range(t - 1, t):  # 初始传入的t为1
            for j in range(0, size):
                fit_list[j][i] = copy.deepcopy(I4fitness_func(X.iloc[j], wd, x_train, x_test, y_train, y_test , t)[0])  # 存放历史粒子的目标值
        k = 1
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
    for i in range(0, size):
        # plt.scatter(list,fit_list[i],c='r')
        cm = plt.cm.get_cmap('RdYlBu')
        sc = plt.scatter(list, fit_list[i], c=list, label=1, vmin=0, vmax=500, cmap=cm)  # 为第i个粒子绘制散点图
        plt.colorbar(sc).remove()
    plt.xlabel("number of iterations")
    plt.ylabel("fitness value")
    # plt.xticks([50,100,150,200,250,300,350,400,450,500])
    plt.legend('')
    plt.axhline(y=0.7, c='b', ls='--', lw='1')
    plt.title(' iterative process')
    plt.show()
    plt.savefig(r"1.png")
    iter_num = 500
    for i in range(1,501):
        list1 += (i,)
    fit_list=pd.DataFrame(columns=list1)
    for k in range(0, iter_num):
        for z in range(4):
            for i in range(0, wd):
                if gbestL[k][z][i] > 0.5:
                    gbestL[k][z][i] = 1
                else:
                    gbestL[k][z][i] = 0
    print(gbestL[iter_num - 1][0])  # 输出最优粒子的特征选择情况(最后一次默认粒子群收敛)
    list1 = []  # 生成存储最优位置重要性的列表
    list2 = []
    list3 = []
    list4 = []
    for i in range(0, wd):  #
        num1 = 0
        num2 = 0
        num3 = 0
        num4 = 0
        for j in range(0, iter_num):
            for z in range(0, 4):
                if z == 0:
                    num1 += gbestL[j][z][i] * gbestL[j][z][wd]
                    list1.append(num1)
                elif z == 1:
                    num2 += gbestL[j][z][i] * gbestL[j][z][wd]
                    list2.append(num2)
                elif z == 2:
                    num3 += gbestL[j][z][i] * gbestL[j][z][wd]
                    list3.append(num3)
                else:
                    num4 += gbestL[j][z][i] * gbestL[j][z][wd]
                    list4.append(num4)

    l1 = gbestL[iter_num - 1][0]
    l2 = gbestL[iter_num - 1][1]
    l3 = gbestL[iter_num - 1][2]
    l4 = gbestL[iter_num - 1][3]
    dt = pd.read_csv(r'd:\test3.csv', encoding='gbk')
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    lu = dt.columns.tolist()
    shengc1 = np.array([np.arange(len(lu)), np.arange(len(lu)), np.arange(len(lu))], dtype=str).T
    shengc2 = np.array([np.arange(len(lu)), np.arange(len(lu)), np.arange(len(lu))], dtype=str).T
    shengc3 = np.array([np.arange(len(lu)), np.arange(len(lu)), np.arange(len(lu))], dtype=str).T
    shengc4 = np.array([np.arange(len(lu)), np.arange(len(lu)), np.arange(len(lu))], dtype=str).T
    for i in range(len(l1) - 1):
        shengc1[i][0] = lu[i]
        shengc1[i][1] = list1[i]
        shengc1[i][2] = l1[i]
    for i in range(len(l1) - 1):
        shengc2[i][0] = lu[i]
        shengc2[i][1] = list1[i]
        shengc2[i][2] = l1[i]
    for i in range(len(l1) - 1):
        shengc3[i][0] = lu[i]
        shengc3[i][1] = list1[i]
        shengc3[i][2] = l1[i]
    for i in range(len(l1) - 1):
        shengc4[i][0] = lu[i]
        shengc4[i][1] = list1[i]
        shengc4[i][2] = l1[i]
    dt1 = pd.DataFrame(data=shengc1, columns=("选择的特征是", "重要度为", "粒子的选择情况是"))
    dt1 = dt1.sort_values(axis=0, by="重要度为", ascending=False)
    dt2 = pd.DataFrame(data=shengc2, columns=("选择的特征是", "重要度为", "粒子的选择情况是"))
    dt2 = dt1.sort_values(axis=0, by="重要度为", ascending=False)
    dt3 = pd.DataFrame(data=shengc3, columns=("选择的特征是", "重要度为", "粒子的选择情况是"))
    dt3 = dt1.sort_values(axis=0, by="重要度为", ascending=False)
    dt4 = pd.DataFrame(data=shengc4, columns=("选择的特征是", "重要度为", "粒子的选择情况是"))
    dt4 = dt1.sort_values(axis=0, by="重要度为", ascending=False)
    dt1.to_csv(r"d:\重要性\重要性排名1.csv")
    dt2.to_csv(r"d:\重要性\重要性排名2.csv")
    dt3.to_csv(r"d:\重要性\重要性排名3.csv")
    dt4.to_csv(r"d:\重要性\重要性排名4.csv")
    # fit_list.to_excel(r'd:\数据\ipso-5.csv')
x_train=pd.read_csv(r'd:\数据\x2_train.csv')
x_test=pd.read_csv(r'd:\数据\x2_test.csv')
y_train=pd.read_csv(r'd:\数据\y2_train.csv')
y_test=pd.read_csv(r'd:\数据\y2_test.csv')
list=x_train.columns.values.tolist()
list1 = ()
iter_num=100
for i in list:
    list1+=(i,)
fit_list = np.zeros(shape=(20, 500))
I4pso(x_train, x_test, y_train, y_test , fit_list, list1)