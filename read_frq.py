#coding=utf-8
import sys, random
sys.path.insert(0, "../../python")
import numpy as np
import codecs
import pandas as pd
import collections
import os
import re

def loadDataSet(path_txt): 
    data_name=[]
    data_label=[]
    with codecs.open(path_txt, "r", encoding="utf-8-sig") as f:
        for line in f:
            lineArr=line.strip().split()
            data_name.append(lineArr[0])
            data_label.append(lineArr[1])
    return data_name,data_label

data_name_test,data_label_test=loadDataSet('test_set.txt')
data_name_train,data_label_train=loadDataSet('train_set.txt')
a=[]
for i in data_label_train:
    a+=i
b=[]
for i in data_label_test:
    b+=i
d_train=collections.Counter(a)
d_test=collections.Counter(b)
df_train = pd.DataFrame.from_dict(d_train, orient='index').reset_index()
df_test = pd.DataFrame.from_dict(d_test, orient='index').reset_index()
df=df_train.merge(df_test,left_on='index', right_on='index',how='outer')
df=df.fillna(0)
df['counts']=df['0_x']+df['0_y']
df=df.drop(['0_x', '0_y'], axis=1)
df['frq']=df.counts/df.counts.sum()
df=df.sort_values(by=['counts'],ascending=False)
df=df.reset_index().drop(['level_0'],axis=1)
df.to_excel('frq.xls')
