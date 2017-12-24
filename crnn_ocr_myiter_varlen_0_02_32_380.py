#coding=utf-8
import sys, random
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx
import codecs
from crnn_model_0_01 import crnn_unroll
import pandas as pd
from io import BytesIO
from captcha.image import ImageCaptcha
import cv2, random
import os
import re



class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


#CHARSET = ''.join([str(i) for i in range(10)])+''.join([chr(i) for i in range(65,91)])+''.join([chr(i) for i in range(97,123)])


def get_label(st):
    global d_get_label
    return [d_get_label[s] if s in d_get_label else 24 for s in st]  #字符里有就贴字符标号，不认识的字就贴37

def get_padded_label(s,cutoff=22):
    out = np.zeros(cutoff)
    beforepad = get_label(s)
    #print len(s),len(out)
    curpos = -1
    for i in range(min(len(s),cutoff)):
        if i>0 and beforepad[i]==24 and beforepad[i-1]==24:  #连续两个不认识的字符统一认为是一个字符
            pass
        else:
            curpos+=1
            out[curpos] = beforepad[i]

        #out[i] = beforepad[i]
    return out

'''
读取标签
'''
DF = pd.read_excel("frqA.xls",sep=',',encoding='utf-8').iloc[:,:5]
d_get_label = {}
d_get_inverse_label = {}
for i in range(24):    #len(DF['index'])
    d_get_label[DF['index'][i]] = i+1
    d_get_inverse_label[i+1] = unicode(DF['index'][i]) 

'''
loadDataSet
input:文件路径
output：
    data_name：文件名list
    data_label：标注list
'''
def loadDataSet(path_txt): 
    data_name=[]
    data_label=[]
    with codecs.open(path_txt, "r", encoding="utf-8-sig") as f:
        for line in f:
            lineArr=line.strip().split()
            data_name.append(lineArr[0])
            data_label.append(lineArr[1])
    return data_name,data_label



class OCRIter(mx.io.DataIter):
    '''
    batch_size: 依次进入的图片大小
    num_label :ctc 切割长度
    init_states:mx传入的初始状态
    path_img:图片文件夹路径
    path_txt:标注文件路径
    白纸图片长380，高32，将图片贴上，在做训练
    
    问题：增广函数还没有写
    '''

    def __init__(self, batch_size, num_label, init_states, path_img,path_txt, check):
        super(OCRIter, self).__init__()
        self.batch_size = batch_size  
        self.num_label = num_label
        self.init_states = init_states
        self.path_img = path_img
        self.path_txt = path_txt
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        #构建一个灰色长条 380*32 然后将图片贴在上面，随后对图片做一些数据曾广，建议数据曾广现在在硬盘里做，当数据足够大后
        #再考虑用mxnet做数据曾广
        
        self.provide_data = [('data', (batch_size,3,32,380))] + init_states  #高32 长380
        self.provide_label = [('label', (self.batch_size, num_label))]
        self.check = check   

    '''
    功能：读取img和label存入data和label的list，data是贴完灰纸的
    '''
    def __iter__(self):
        #print 'iter'
        init_state_names = [x[0] for x in self.init_states]
        
        data_name,data_label=loadDataSet(self.path_txt)
        pic_num = len(data_name)
        num = 0
        for k in range(pic_num / self.batch_size):
            data = []
            #data_name_list=[]
            label=[]
            
            for i in range(self.batch_size):
                if num > pic_num-1:
                    break
                while 1:
                    #print path_img +'/'+ data_name[num]  
                    res = cv2.imread(self.path_img +'/'+ data_name[num])#脑残用了dir_list 
                    
                    if res.any() != None:
                        break    
                #rand_size=random.uniform(0.9,1.1)
                #try:
                    #res=cv2.resize(img,(int(rand_size*img.shape[1]),int(rand_size*img.shape[0])),interpolation=cv2.INTER_CUBIC)
                #except:
                    #res=img
                # if res.shape[0]>32:
                #     res=res[np.absolute(res.shape[0]-32):32, :]   
                if res.shape[1]>380:
                    res=cv2.resize(res, (380, 32)) #这段竟然没整出来   正常来讲应该删除但是num不知道怎么控制了)    

                       
                #data_name_list.append(data_name[num])   #保留看名字
                label.append(get_padded_label(data_label[num]))
                num += 1#
                '''
                数据增广：上下左右随机偏移0-3个像素
                '''
                newimg = np.zeros((32, 380,3))+int(res.mean())



                rand_1=random.randint(-3,3)
                rand_2=int((380-res.shape[1])*random.random())
                

                for n in range(rand_1,res.shape[0]):     # 千万不能用i  低级错误
                    if n-rand_1>=res.shape[0]:
                        break
                    for m in range(rand_2,rand_2+res.shape[1]):
                        if m-rand_2>=res.shape[1]:
                            break            
                        newimg[n-rand_1,m] = res[n-rand_1,m-rand_2]
                
                newimg = newimg.reshape((3,32 , 380)) 
                newimg = np.float32(np.multiply(newimg, 1/255.0))
                data.append(newimg)
                
            
            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['label']
            
            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass




BATCH_SIZE = 160
SEQ_LENGTH = 22

def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i+1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret


def remove_blank(l):
    ret = []
    for i in range(len(l)):
        if l[i] == 0:
            break
        ret.append(l[i])
    return ret

def remove_trailing_zero(l):
    i = 0
    #print l
    for i in range(len(l)-1):
        if l[-i-1]==0 and l[-i-2]==0: 
            continue
        else:
            break
    if i == 0:
        return l
    else:
        return l[:-i]


def Accuracy(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = label[i]
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        if len(p) == len(l):
            match = True
            for k in range(len(p)):
                if p[k] != int(l[k]):
                    match = False
                    break
            if match:
                hit += 1.0
        total += 1.0
    return hit / total

def LCS(p,l):
    if len(p)==0 or len(l)==0:
        return 0
    P = np.array(list(p)).reshape((1,len(p)))
    L = np.array(list(l)).reshape((len(l),1))
    M = np.int32(P==L)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            up = 0 if i==0 else M[i-1,j]
            left = 0 if j==0 else M[i,j-1]
            M[i,j] = max(up,left, M[i,j] if (i==0 or j==0) else M[i,j]+M[i-1,j-1])
    return M.max()

def LCS_Purge(p,l):
    mp = [item for item in p if item!=25]
    ml = [item for item in l if item!=25]
    if len(mp)==0 or len(ml)==0:
        return 0
    P = np.array(list(mp)).reshape((1,len(mp)))
    L = np.array(list(ml)).reshape((len(ml),1))
    M = np.int32(P==L)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            up = 0 if i==0 else M[i-1,j]
            left = 0 if j==0 else M[i,j-1]
            M[i,j] = max(up,left, M[i,j] if (i==0 or j==0) else M[i,j]+M[i-1,j-1])
    return M.max()


def Accuracy_LCS(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    global CTX_LENGTH
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        
        ## Dynamic Programming Finding LCS
        if len(l)==0:
            continue
        hit += LCS(p,l)*1.0/len(l)
        total += 1.0
    return hit / total

def Accuracy_LCS_Purge(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    global CTX_LENGTH
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        
        ## Dynamic Programming Finding LCS
        ml = [item for item in l if item!=37]
        ml = remove_trailing_zero(ml)
        if len(ml)==0:
            pass
        else:
            hit += LCS_Purge(p,ml)*1.0/len(ml)
            total += 1.0
    if total == 0:
        print "Total:0"
        return 0
    else:
        return hit / total


    global BATCH_SIZE
    global SEQ_LENGTH
    global CTX_LENGTH
    hit = 0.
    total = 0.
    
    hithan = 0.
    totalhan = 0.
    hitnon = 0.
    totalnon = 0.
    
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        
        ## Dynamic Programming Finding LCS
        ml = [item for item in l if item!=24]
        ml = remove_trailing_zero(ml)
        if len(ml)==0:
            pass
        else:
            hit += LCS_Purge(p,ml)*1.0/len(ml)
            total += 1.0
            
        ## Han LCS
        phan = [item for item in p if isLabelHan(item)]
        mlhan = [item for item in ml if isLabelHan(item)]
        
        if len(mlhan)==0:
            pass
        else:
            hithan += LCS_Purge(phan, mlhan)*1.0/len(mlhan)
            totalhan += 1.0
        
        ## Non-Han LCS
        pnon = [item for item in p if not isLabelHan(item)]
        mlnon = [item for item in ml if not isLabelHan(item)]
        
        if len(mlnon)==0:
            pass
        else:
            hitnon += LCS_Purge(pnon, mlnon)*1.0/len(mlnon)
            totalnon += 1.0
        
        #print ml
        #print mlhan
        #print mlnon
        #break
        
    if totalhan==0:
        print "totalhan:0"
        #return 0
    else:
        return hithan / totalhan if totalhan > 0.0 else 0 
        #return hitnon / totalnon if totalnon > 0.0 else 0
        #return  hit / total
        #                 (hithan / totalhan if totalhan > 0.0 else 0 ),
        #                 (hitnon / totalnon if totalnon > 0.0 else 0 )]
                         


    

if __name__ == '__main__':
    BATCH_SIZE = 160
    SEQ_LENGTH = 22
    num_hidden = 512  
    num_lstm_layer = 2
    num_epoch = 1000
    learning_rate = 0.0005
    momentum = 0.9
    num_label = 22

    # 256*2, lr = 0.002 Best Ever
    
    prefix = 'mymodel/CRNN_BN_512_1024_1024_1024_ADADELTAMIXLANGUAGE_BATCH{bat}_SEQ{seq}_varlen_{hid}-{nlay}-{lr}-100-mmt{mmt}-pred'.\
            format(bat=BATCH_SIZE,seq=SEQ_LENGTH,hid=num_hidden,nlay=num_lstm_layer,lr=learning_rate,mmt=momentum)
    iteration = 10
    #devices = [mx.gpu(i) for i in range(2)]
    contexts = [mx.context.gpu(0)]
    
#    print prefix

    def sym_gen(seq_len):
        return crnn_unroll(num_lstm_layer, seq_len,
                           num_hidden= num_hidden,
                           num_label = num_label)

    init_c = [('l%d_init_c'%l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    img_path  = 'images'
    train_txt = 'train_setA.txt' 
    test_txt  = 'test_setA.txt'
    data_train = OCRIter(BATCH_SIZE, num_label, init_states, img_path,train_txt, 'train')
    data_val =   OCRIter(BATCH_SIZE, num_label, init_states, img_path,test_txt, 'test')
    
   


    symbol = sym_gen(SEQ_LENGTH)

    model = mx.model.FeedForward(ctx=contexts,
                                 
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 #momentum=momentum,
                                 optimizer='adadelta',
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
    #model.bind(for_training=False, data_shapes=[('data', (BATCH_SIZE,3,32,380))])
    
    # load model
    # prefix_load='CRNN_BN_512_1024_1024_1024_ADADELTAMIXLANGUAGE_BATCH128_SEQ22_varlen_1024-2-0.0003-100-mmt0.9-predhan-symbol.json'
    # model = mx.model.FeedForward.load(prefix,iteration,
    #                                 learning_rate = learning_rate,
    #                                 ctx = contexts,
    #                                 numpy_batch_size = BATCH_SIZE,
    #                                 num_epoch=num_epoch)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.basicConfig(filename="gpu0_256.log",level=logging.DEBUG, format=head)
    

    print 'begin fit'
    
    model.fit(X=data_train, eval_data=data_val,
              eval_metric = mx.metric.np(Accuracy_LCS),
              batch_end_callback = mx.callback.Speedometer(BATCH_SIZE, 150),
              epoch_end_callback = mx.callback.do_checkpoint(prefix, 1))
             #)
    model.save(prefix, iteration)
