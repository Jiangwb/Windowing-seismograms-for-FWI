# -*- coding: utf-8 -*-
"""
Created on May 14 2018

"""
import os 
import numpy as np
from keras.models import *
from keras.layers import Input, merge, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras import losses
from keras.models import load_model
from keras.utils import np_utils
from keras.models import Sequential
import matplotlib.pyplot as plt
import math
from preprocess import mkdir,preprocessdata
from scipy.optimize import curve_fit
import scipy.io as sio
import time


def write2txt(labelcontent,num,samplea,sampleb,samplec):
    fp=open(r'./test/Log.txt','w')
    for lat in range(samplea):
        for lon in range(sampleb):
            for dep in range(samplec):
                content=str(labelcontent[lat,lon,dep])+'\r\n'
                fp.write(content)
    fp.close()
    
def f_gauss2(x, mu, r,C):
    return (1/(2*math.pi)**(0.5)/r)*np.exp(-((x*0.1-mu)**2)/(2*r**2))+C

def GaussFitting(y):

    x = np.arange((len(y)))
    # 高斯曲线拟合
    mu, r,C= curve_fit(f_gauss2, x, y)[0]
    # 生成拟合Gauss Curve
    y = (1/(2*math.pi)**(0.5)/r)*np.exp(-((x*0.1-mu)**2)/(2*r**2))+C
    
    pos=np.where(y == np.max(y))
    return  np.array(pos,dtype=float)

def loadModel(model_load_path):
    #------------------------Load Model -----------------------------
    # 建立序贯模型
    model = Sequential()
    # returns a compiled model
    # identical to the previous one
    # deletes the existing model
    del model  
    model = load_model(model_load_path)
    model.summary()
    # 配置模型的学习过程
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model

def p1(para1):
    latmin,lat_interval = 28.0,0.1   
    para1_out = latmin + lat_interval*para1
    return round(para1_out,2) 

def p2(para2):
    lonin,lon_interval = 100.6,0.1  
    para2_out = lonin + lon_interval*para2
    return round(para2_out,2)

    
def show_label(label,label_pred,quene,fig_path=''):
    if fig_path is not '':
        flag = True
    else:
        flag = False
    if quene==[]:
        quene = np.arange(label.shape[0])
   
    for i in quene:
        pred_starttime = time.time()
        onelabel_pred = label_pred[i,:,0]
        pred_endtime = time.time()
        time_pred = pred_endtime - pred_starttime
        print('time_pred',time_pred)
        # onelabel = label[i,:,0]
        
        fig = plt.figure()  
        
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
        
        ax1.plot(onelabel)
        ax2.plot(onelabel_pred)
        if flag:
            print('output figure: %d \n' % i)
            fig.savefig(fig_path+'/'+str(i))
        #plt.show()
    
def show_data_and_label(data,label_pred,quene,fig_path='',label_predpath=''):
    if fig_path is not '':
        flag = True
    else:
        flag = False
    if quene==[]:
        quene = np.arange(label_pred.shape[0])
    
    for i in quene:
        onetrace_Z = data[i,:,0]
        pred_starttime = time.time()
        onelabel_pred = label_pred[i,:,0]
        pred_endtime = time.time()
        time_pred = pred_endtime - pred_starttime
        print('time_pred',time_pred)
        # onelabel = label[i,:,0]
        
        fig = plt.figure()  
        
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
        #ax1.imshow(onelabel)
        #ax2.imshow(onelabel_pred)

        ax1.plot(onetrace_Z)
        ax2.plot(onelabel_pred)
        if flag:
            print('output figure: %d \n' % i)
            fig.savefig(fig_path+'/'+str(i))
            
##################xi_chengpeng add###################            
def save_label_pred(result_load_path,data,label_pred,label_predpath):
    import scipy.io as sio
    index = np.arange(label_pred.shape[0])
    x = np.zeros(4992)
    
    for j in index:
        x = label_pred[j][:]
        sio.savemat(label_predpath + '/' + str(j) + '.mat', {'x':x})
    # from obspy import Trace
    # index = np.arange(label_pred.shape[0])
    # # x = np.zeros(4992) 
    # for j in index:
    #     x = np.zeros(4992)
    #     for z in range(len(label_pred[0])):
    #         x[z] = label_pred[j][z][0]
    #         sacfile = Trace()
    #         sacfile.data = x[:]
    #         sacfile.write(label_predpath +'/' + str(j) + ".sac",format = "SAC")    
    #     print('label' + str(j))
##################xi_chengpeng add################### 

def savemat(data_load_path,testnum,data):
    imgs_test = data
    model_load_path = r'./test/unet.hdf5'
    model = loadModel(model_load_path)
    inter_layer = Model(inputs=model.input, outputs=model.get_layer('conv1d_2').output).predict(x=(imgs_test))
    sio.savemat('layer1.mat', {'lay1':inter_layer})
    print('finished  layer1.mat')
    inter_layer = Model(inputs=model.input, outputs=model.get_layer('conv1d_4').output).predict(x=imgs_test)
    sio.savemat('layer2.mat', {'lay2':inter_layer})
    print('finished  layer2.mat')
    inter_layer = Model(inputs=model.input, outputs=model.get_layer('conv1d_6').output).predict(x=imgs_test)
    sio.savemat('layer3.mat', {'lay3':inter_layer})
    print('finished  layer3.mat')
    inter_layer = Model(inputs=model.input, outputs=model.get_layer('conv1d_8').output).predict(x=imgs_test)
    sio.savemat('layer4.mat', {'lay4':inter_layer})
    print('finished  layer4.mat')
    inter_layer = Model(inputs=model.input, outputs=model.get_layer('conv1d_10').output).predict(x=imgs_test)
    sio.savemat('layer5.mat', {'lay5':inter_layer})
    print('finished  layer5.mat')
    inter_layer = Model(inputs=model.input, outputs=model.get_layer('conv1d_12').output).predict(x=imgs_test)
    sio.savemat('layer6.mat', {'lay6':inter_layer})
    print('finished  layer6.mat')
    inter_layer = Model(inputs=model.input, outputs=model.get_layer('conv1d_15').output).predict(x=imgs_test)
    sio.savemat('layer7.mat', {'lay7':inter_layer})
    print('finished  layer7.mat')
    inter_layer = Model(inputs=model.input, outputs=model.get_layer('conv1d_18').output).predict(x=imgs_test)
    sio.savemat('layer8.mat', {'lay8':inter_layer})
    print('finished  layer8.mat')
    inter_layer = Model(inputs=model.input, outputs=model.get_layer('conv1d_21').output).predict(x=imgs_test)
    sio.savemat('layer9.mat', {'lay9':inter_layer})
    print('finished  layer9.mat')
    inter_layer = Model(inputs=model.input, outputs=model.get_layer('conv1d_24').output).predict(x=imgs_test)
    sio.savemat('layer10.mat', {'lay10':inter_layer})
    print('finished  layer10.mat')
    inter_layer = Model(inputs=model.input, outputs=model.get_layer('conv1d_27').output).predict(x=imgs_test)
    sio.savemat('layer11.mat', {'lay11':inter_layer})
    print('finished  layer11.mat')
    inter_layer = Model(inputs=model.input, outputs=model.get_layer('conv1d_30').output).predict(x=imgs_test)
    sio.savemat('layer12.mat', {'lay12':inter_layer})
    print('finished  layer12.mat')
    inter_layer = Model(inputs=model.input, outputs=model.get_layer('conv1d_33').output).predict(x=imgs_test)
    sio.savemat('layer13.mat', {'lay13':inter_layer})
    print('finished  layer13.mat')
    inter_layer = Model(inputs=model.input, outputs=model.get_layer('conv1d_37').output).predict(x=imgs_test)
    sio.savemat('layer14.mat', {'lay14':inter_layer})
    print('finished  layer14.mat')
    inter_layer = Model(inputs=model.input, outputs=model.get_layer('conv1d_38').output).predict(x=imgs_test)
    sio.savemat('layer15.mat', {'lay15':inter_layer})
    print('finished  layer15.mat')
        
if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    # list parameters
    parameter1,parameter2 = 'Latitude','Longitude'
    # data file path  
    data_load_path = [r'../data/predict']
    # trainded model path 
    model_load_path = r'./test/unet.hdf5'
    # fig_path
    fig_path = r'./out_put/figure'
    mkdir(fig_path)
    
    from fcn_test import myFcn
    myfcn = myFcn(component = 1)
    model = myfcn.get_fcn()
    # result file path
    #label_predpath = r'../data/predict/conv_pre/test'
    label_predpath = r'./out_put'
    mkdir(label_predpath)
    result_load_path = r'./out_put'
    mkdir(result_load_path)
    # the number of train and test
    trainnum,testnum = 0, 300
    # the length of samples of one trace,the components of the data
    img_cols,component = 4992,1
    # load data       
    data, label = preprocessdata(data_load_path,trainnum,testnum,img_cols,component,'predict')
    # load model
    model = loadModel(model_load_path)
    # predict label 
    label_pred = model.predict(data, batch_size=32, verbose=0)
    savemat(data_load_path,testnum,data)
    # save result 
    save_label_pred(result_load_path,data,label_pred,label_predpath)
    # show label
    #show_label(label,label_pred,[],fig_path)
    #show_data_and_label(data,label_pred,[],fig_path)
