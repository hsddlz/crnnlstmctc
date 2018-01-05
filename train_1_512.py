#coding:utf-8

import numpy as np
import random
import os
import cv2
import codecs
import pandas as pd
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

'''
提取文件名与label
input:txt标注文件  20171220_54775_typeA.jpg 3+10=13 回车  前端是文件名 后端是label
output：[[img1][img2].....]  [[label1][label2]....]  
'''
def loadDataSet(path_txt): 
    JZdic={u'=':u'=',u'﹦':u'=',u'﹦':u'=',u'0':u'0',u'1':u'1',u'2':u'2',u'3':u'3',u'4':u'4',u'5':u'5',u'6':u'6',u'7':u'7',u'8':u'8',u'9':u'9',u'+':u'+',u'-':u'-',u'×':u'×',u'x':u'×',
       u'*':u'×',u'÷':u'÷',u'.':u'.',u'｛':u'{',u'｝':u'}',u')':u')',u'(':u'(',u'＋':u'+',u'－':u'-',u'_':u'-',u'[':u'[',u']':u']',u'<':u'<',u'＜':u'<',
       u'＞':u'>',u'>':u'>',u'（':u'(',u'＝':u'=',u'﹤':u'<',u'﹥':u'>',u'）':u')',u'］':u']',u'［':u'[',u'@':u'@',u'＠':u'@'}
    data_name=[]
    data_label=[]
    with codecs.open(path_txt, "r", encoding="utf-8-sig",) as f:
        for line in f:
            lineArr=line.strip().split()
            for word, initial in JZdic.items():
                try:
                    lineArr[1] = lineArr[1].replace(word, initial)
                except:
                    #print line
                    continue
            try:
                data_name.append(lineArr[0])
            except:
                continue
            data_label.append(lineArr[1])

    print len(data_name),len(data_label)  #洗前      
    err_name=[]
    err_label=[]
    for i in range(len(data_label)):
        img=cv2.imread('images/'+data_name[i])
        try:
            rate=img.shape[1]/img.shape[0]
        except:
            print data_name[i]
        sym_A=data_label[i].count(u'=',0,len(data_label[i]))>=2 or data_label[i].count(u'++',0,len(data_label[i]))>0 or data_label[i].count(u'--',0,len(data_label[i]))>0 or data_label[i].count(u'××',0,len(data_label[i]))>0 or data_label[i].count(u'÷÷',0,len(data_label[i]))>0
        if len(data_label[i])>9 and sym_A and rate<3 :
            #print data_label[i]
            err_name.append(data_name[i])
            err_label.append(data_label[i])
    data=pd.DataFrame(list(zip(data_name,data_label)),columns=['name','label'])
    data_name=data[~data.name.isin(err_name)].name.values.tolist()
    data_label=data[~data.name.isin(err_name)].label.values.tolist()
    print len(data_name),len(data_label) #洗后
    return data_name,data_label

def preprocess(fnames, labels,area):
    global height, width, n
    
    shapes = [cv2.imread('images/'+x).shape for x in fnames]

    shapes = np.array(shapes)

    index_useful = np.argsort(shapes[:,1])[area:-area]   #排序，干掉最大最小的

    np.random.shuffle(index_useful)

    fnames = [fnames[x] for x in index_useful]
    labels = [labels[x] for x in index_useful]    
    index_useful = np.argsort([len(x) for x in labels])[area:-area]
    fnames = [fnames[x] for x in index_useful]
    labels = [labels[x] for x in index_useful]  
    length = [cv2.imread('images/'+x).shape[1] for x in fnames]
    max_length = np.max(length)
    print 'maxwidth='+str(max_length)
    return fnames, labels,max_length

def pro_do(txtA,txtB,n_class =27,n_len = 16,width=275,height=32):
    
    fnames = []
    labels = []
    a, b = loadDataSet(txtA)
    fnames += a
    labels += b
    fnames, labels,max_lengthA = preprocess(fnames, labels,3000) #数据集A效果不太好，多干掉点

    n = len(fnames)
    random_index = range(n)
    np.random.shuffle(random_index)
    fnames = [fnames[x] for x in random_index]
    labels = [labels[x] for x in random_index]

    X_t, X_v, y_t, y_v = train_test_split(fnames, labels, test_size=0.2)

    c, d = loadDataSet(txtB)
    X_t += c
    y_t += d
    X_t, y_t,max_lengthB = preprocess(X_t, y_t,4000)
    width =np.max([max_lengthA,max_lengthB])
    n_train=len(X_t)
    n_val=len(X_v)
    
    from collections import Counter
    c = Counter(''.join(y_t+y_v))
    characters = ''.join([x[0] for x in c.most_common()][:26]) + ' '
    print characters

    print n_train , n_val , width  # 输出最后训练长度，测试长度，图片最大宽度
#生成训练集
    X_train = np.zeros((n_train, width, height, 1), dtype=np.uint8)
    y_train = np.ones((n_train, n_len), dtype=np.int64) * (n_class-1)
    for i in tqdm(range(n_train)):
        j = 0
        for c in y_t[i]:
            if j >=n_len:    #大于16后面的字符就不要了
                
                break
            if c in characters:
                y_train[i, j] = characters.index(c)
                j += 1

        img = cv2.imread('images/'+X_t[i], 0)
        X_train[i, :img.shape[1], :img.shape[0], 0] = img.transpose(1, 0)
#生成测试集
    X_val = np.zeros((n_val, width, height, 1), dtype=np.uint8)
    y_val = np.ones((n_val, n_len), dtype=np.int64) * (n_class-1)
    for i in tqdm(range(n_val)):    
        j = 0
        for c in y_v[i]:
            if j >=n_len:
                
                break
            if c in characters:
                y_val[i, j] = characters.index(c)
                j += 1
        img = cv2.imread('images/'+X_v[i], 0)
        X_val[i, :img.shape[1], :img.shape[0], 0] = img.transpose(1, 0)



    df=pd.DataFrame(X_v,columns=['name'])
    df['label']=y_v
    df.to_excel('typeA.xls')  #保留测试集  为后来找badcase用
    print n_train,n_val  #训练集测试集长度  
    return X_train,y_train,X_val,y_val,n_train,n_val,characters,width


A='typeA.txt'
B='typeB.txt'
X_train,y_train,X_val,y_val,n_train,n_val,characters,width=pro_do(A,B)
width,height=width,32
n_class =27
n_len = 16

from keras import backend as K

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def evaluate():
    y_pred = base_model.predict(X_val, batch_size=256)
    shape = y_pred[:,2:,:].shape
    out = K.get_value(K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0])[:, :n_len]
    if out.shape[1] > 10:
        out[out < 0] = n_class - 1
        return (y_val[:,:out.shape[1]] == out).all(axis=-1).mean()
    return 0

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2

rnn_size = 512
l2_rate = 1e-5

input_tensor = Input((width, height, 1))
x = input_tensor
x = Lambda(lambda x:(x-127.5)/127.5)(x)
for i, n_cnn in enumerate([3, 4, 6]):
    for j in range(n_cnn):
        x = Conv2D(32*2**i, (3, 3), padding='same', kernel_initializer='he_uniform', 
                   kernel_regularizer=l2(l2_rate))(x)
        x = BatchNormalization(gamma_regularizer=l2(l2_rate), beta_regularizer=l2(l2_rate))(x)
        x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

cnn_model = Model(input_tensor, x, name='cnn')

input_tensor = Input((width, height, 1))
x = cnn_model(input_tensor)

conv_shape = x.get_shape().as_list()
rnn_length = conv_shape[1]
rnn_dimen = conv_shape[3]*conv_shape[2]

print conv_shape, rnn_length, rnn_dimen

x = Reshape(target_shape=(rnn_length, rnn_dimen))(x)
rnn_length -= 2
rnn_imp = 1

x = Dense(rnn_size, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate))(x)
x = BatchNormalization(gamma_regularizer=l2(l2_rate), beta_regularizer=l2(l2_rate))(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)

gru_1 = GRU(rnn_size, implementation=rnn_imp, return_sequences=True, name='gru1')(x)
gru_1b = GRU(rnn_size, implementation=rnn_imp, return_sequences=True, go_backwards=True, name='gru1_b')(x)
gru1_merged = add([gru_1, gru_1b])

gru_2 = GRU(rnn_size, implementation=rnn_imp, return_sequences=True, name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, implementation=rnn_imp, return_sequences=True, go_backwards=True, name='gru2_b')(gru1_merged)
x = concatenate([gru_2, gru_2b])

x = Dropout(0.2)(x)
x = Dense(n_class, activation='softmax', kernel_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate))(x)
rnn_out = x
base_model = Model(input_tensor, x)

label_tensor = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), 
                  name='ctc')([x, label_tensor, input_length, label_length])

model = Model(inputs=[input_tensor, label_tensor, input_length, label_length], outputs=[loss_out])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
#model=load_model('model.h5')

from keras.utils import multi_gpu_model
model_parallel = multi_gpu_model(model, 2)

model_parallel.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(1e-4))

from keras.callbacks import *

class Evaluate(Callback):
    def __init__(self):
        self.accs = []
    
    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate()*100
        self.accs.append(acc)
        print '   '
        print '   '
        print 'val_acc: %f%%'%acc

evaluator = Evaluate()



model_parallel.fit([X_train, y_train, np.ones(n_train)*rnn_length, np.ones(n_train)*n_len], np.ones(n_train), 
                   validation_data=([X_val, y_val, np.ones(n_val)*rnn_length, np.ones(n_val)*n_len], np.ones(n_val)), 
                   callbacks=[evaluator], 
                   batch_size=400, epochs=25)

model_parallel.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(1e-5))
h = model_parallel.fit([X_train, y_train, np.ones(n_train)*rnn_length, np.ones(n_train)*n_len], np.ones(n_train), 
                       validation_data=([X_val, y_val, np.ones(n_val)*rnn_length, np.ones(n_val)*n_len], np.ones(n_val)), 
                       callbacks=[evaluator], 
                       batch_size=400, epochs=15)
model_parallel.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(1e-6))
h = model_parallel.fit([X_train, y_train, np.ones(n_train)*rnn_length, np.ones(n_train)*n_len], np.ones(n_train), 
                       validation_data=([X_val, y_val, np.ones(n_val)*rnn_length, np.ones(n_val)*n_len], np.ones(n_val)), 
                       callbacks=[evaluator], 
                       batch_size=400, epochs=4)
base_model.save('model.h5')




# n_test=len(X_val)
# y_pred=[]    

# X_test = np.zeros((n_test, width, height, 1), dtype=np.uint8)

# y_p = base_model.predict(X_val)
# shape = y_p[:,2:,:].shape
# out = K.get_value(K.ctc_decode(y_p[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0])[:, :n_len]
# pred_label=[]
# for i,n in enumerate(out):
#     s = ''.join([characters[x] for x in out[i] if x > -1])
#     pred_label.append(s)
# pdf=pd.DataFrame(list(zip(y_test,pred_label)),columns=['name','y','pred'])
# pdf.to_excel('dd.xls')
