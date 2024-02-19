import pdb

import pandas as pd
# data1=pd.read_csv(r'd:\train\cat-f1.csv')
# data2=pd.read_csv(r'd:\train\lgb-f1.csv')
# data3=pd.read_csv(r'd:\train\xgb-f1.csv')
# feature=data1['feature'].tolist()
# score=[]
# for i in feature:
#     s=0
#     s+=data1[data1['feature']==i]['score'].iloc[0]
#     # pdb.set_trace()
#     s += data2[data2['feature'] == i]['score'].iloc[0]
#     s += data3[data3['feature'] == i]['score'].iloc[0]
#     score.append(s)
# datanew=pd.DataFrame(columns=('feature','score'))
# datanew['feature']=feature
# datanew['score']=score
# datanew.to_csv(r'd:\train\acc-result',index=0)
data=pd.read_csv(r'd:\train\acc-result')
dataf=pd.read_csv(r'd:\train\1-4-29.txt')
# pdb.set_trace()
data=data.sort_values(by='score',ascending=False)
data1=data.iloc[:40,:]
feature=data1['feature'].tolist()
feature1=dataf.columns.tolist()[:-1]
# pdb.set_trace()
for i in feature1:
    if i not in feature and i !='flag':
        dataf=dataf.drop(i,axis=1)
pdb.set_trace()