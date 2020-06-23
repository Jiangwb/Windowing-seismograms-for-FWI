# -*- coding: utf-8 -*-
"""
Created on May 11 2019

"""
import time
import os 
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats
import math
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt # plt for display
import ctypes
from obspy import read
import keras.backend.tensorflow_backend as KTF

#config = tf.ConfigProto()
#sess = tf.Session(config=config)
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#config.gpu_options.per_process_gpu_memory_fraction = 0.9

def mkdir(path):
    # check if path exist
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        print (path+'Creating success')
        return True
    else:
        print (path+'Path already exist')
        return False
    
def preprocessdata(filepath,trainnum,testnum,img_cols,component,flag='train'):
    
    eventnum = trainnum + testnum
    data=np.empty((eventnum,img_cols,component))
    label=np.empty((eventnum,img_cols,1))
    
    comp=['Z','R','T']
    k=0
	####### convert list to string #######
    list2=[str(i) for i in filepath]
    list3=''.join(list2)
	####### convert list to string #######
    file = list3+'/syn/Z'
    catalog = os.listdir(file) # all the trace is Z component
    catalog.sort(key=lambda x:int(x[:-4]))
    # catalog.sort = []
    print('scrambling the order of the files')
    # print(len(catalog))
    newcatalog = []
    index=[i for i in range(len(catalog))]
    random.seed(15)
    random.shuffle(index)
    for i in index :
        newcatalog.append(catalog[i])
    for event in newcatalog:
         start = time.time()
         sorted(event)
         if k>=eventnum:
            break
         if flag == 'predict':    
            judge_state = k>=0 and k<testnum
         elif flag == 'train': 
              judge_state = k>=0 and k<eventnum
         else:
            print('flag state is wrong! Please input predict or train')
            return
         if judge_state:
            print('Processing No.'+str(k)+' '+event) 
			# read three component data
            for chan in range(component):
                eventname = event.replace('LHZ','LH'+comp[chan])
                print(eventname)
                if not os.path.exists(list3+'/syn'+'/'+comp[chan]+'/'+eventname):
                    print(' No eventname'+eventname)
                    continue
                data_path=list3+'/'+'syn'+'/'+comp[chan]+'/'+eventname
                temp = read(data_path,debug_headers=True)
                data[k,:,chan] = temp[0].data[0:]
			# read label data
            labelname = event.replace('LHZ','LH'+comp[chan])
            if not os.path.exists(list3+'/label'+'/'+labelname):
                print(' No labelname'+labelname)
                continue
            data_path=list3+'/'+'label'+'/'+labelname
            temp = read(data_path,debug_headers=True)
            label[k,:,0] = temp[0].data[0:]
            k=k+1
    if flag == 'predict':
        test_data = data[0:testnum,:]
        test_label = label[0:testnum,:]
        return  test_data, test_label
    else: 
        train_data = data[0:trainnum,:]
        train_label = label[0:trainnum,:]
        test_data = data[trainnum:,:]
        test_label = label[trainnum:,:]
        return  train_data, train_label, test_data, test_label
    

if __name__ == '__main__':
    
    filepath=[r'./data/train'] 
    trainnum,testnum = 750,259
    img_cols = 4992    # the length of samples of one trace
    component = 1     # the components of the data
    train_data, train_label, test_data, test_label = preprocessdata(filepath,trainnum,testnum,img_cols,component)
