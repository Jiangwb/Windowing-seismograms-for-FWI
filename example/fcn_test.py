# -*- coding: utf-8 -*-

"""
Created on May 12 2019

@author: Wenbin Jiang; Xi_chengpeng
"""
from keras.utils import multi_gpu_model
import tensorflow as tf
import keras
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from preprocess import mkdir,preprocessdata
from keras.callbacks import ModelCheckpoint
import os
import keras.models
from keras.models import Sequential
from keras.layers import Input, merge, concatenate, Conv1D, MaxPooling1D, UpSampling1D, Dropout, Cropping1D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,Callback
from keras import losses
import numpy as np
from keras.models import *
from keras.optimizers import *
from keras.callbacks import Callback
from keras.utils import np_utils
import matplotlib.image as mpimg # mpimg read figure
import scipy.misc
import keras.backend.tensorflow_backend as KTF

#config = tf.compat.v1.ConfigProto()
#sess = tf.compat.v1.Session(config=config)
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

#session = InteractiveSession(config=config)

class LossHistory(keras.callbacks.Callback):

	def on_train_begin(self, logs={}):
		self.losses = {'batch':[], 'epoch':[]}
		self.accuracy = {'batch':[], 'epoch':[]}
		self.val_loss = {'batch':[], 'epoch':[]}
		self.val_acc = {'batch':[], 'epoch':[]}

	def on_batch_end(self, batch, logs={}):
		self.losses['batch'].append(logs.get('loss'))
		self.accuracy['batch'].append(logs.get('acc'))
		self.val_loss['batch'].append(logs.get('val_loss'))
		self.val_acc['batch'].append(logs.get('val_acc'))

	def on_epoch_end(self, batch, logs={}):
		self.losses['epoch'].append(logs.get('loss'))
		self.accuracy['epoch'].append(logs.get('acc'))
		self.val_loss['epoch'].append(logs.get('val_loss'))
		self.val_acc['epoch'].append(logs.get('val_acc'))

	def loss_plot(self, loss_type, savepath):
		iters = range(len(self.losses[loss_type]))
		plt.figure()
        # acc
#		plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
		plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
		if loss_type == 'epoch':
			# val_acc
#			plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
			# val_loss
			plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
		plt.grid(True)
		plt.xlabel(loss_type)
		plt.ylabel("Loss Function")
		plt.legend(loc="upper right")
		fig = plt.gcf()
		plt.show()
		fig.savefig(savepath)

	def write_log(self,log_path,score):
		#a1 = self.accuracy['epoch']
		a2 = self.losses['epoch']
		#a3 = self.val_acc['epoch']

		a4 = self.val_loss['epoch']
		with open(log_path,"w") as f:
			curves_data=[]
			f.write("train-loss \t val-loss \n")
			for i in range(len(a2)):
				curves_data.append([a2[i],a4[i]])
			for i in range(len(a2)):
				temp = [str(j) for j in curves_data[i]]
				f.write('\t\t'.join(temp)+'\n')
			#f.write('\nTest score:\t\n',  (score[0]))
			#f.write('Test accuracy:\t', (100*score[1]))

class myFcn(object):

    def __init__(self, img_cols = 4992, component = 1):

        self.img_cols = img_cols
        self.component = component
    def get_fcn(self):

		#inputs = Input((self.img_cols,self.component,1))
        inputs = Input((self.img_cols,self.component))
        down1 = 2
        down2 = 2
        down3 = 2
        down4 = 2
        down5 = 2
        down6 = 2
        down7 = 2
        down8 = 2

        up9 = 2
        up10 = 2
        up11 = 2
        up12 = 2
        up13 = 2
        up14 = 2
        up15 = 2

        conv1 = Conv1D(4, kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv1D(4, kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling1D(pool_size=down1, padding = 'same')(conv1)

        conv2 = Conv1D(8,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv1D(8,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling1D(pool_size=down2, padding = 'same')(conv2)

        conv3 = Conv1D(16,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv1D(16,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling1D(pool_size=down3, padding = 'same')(conv3)

        conv4 = Conv1D(32,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv1D(32,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling1D(pool_size=down4, padding = 'same')(drop4)

        conv5 = Conv1D(64,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv1D(64,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
        pool5 = MaxPooling1D(pool_size=down5,padding = 'same')(drop5)

        conv6 = Conv1D(128,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
        conv6 = Conv1D(128,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        drop6 = Dropout(0.5)(conv6)
        pool6 = MaxPooling1D(pool_size=down6, padding = 'same')(drop6)

        conv7 = Conv1D(256,kernel_size=3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool6)
        conv7 = Conv1D(256,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        drop7 = Dropout(0.5)(conv7)
        pool7 = MaxPooling1D(pool_size=down7,padding = 'same')(drop7)

        conv8 = Conv1D(512,kernel_size=3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool7)
        conv8 = Conv1D(512,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        drop8 = Dropout(0.5)(conv8)

        up9 = Conv1D(256,kernel_size=3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = up9)(conv8))
        conv9 = Conv1D(256, kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
        conv9 = Conv1D(256, kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        up10 = Conv1D(128,kernel_size=3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = up10)(conv9))
        conv10 = Conv1D(128, kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up10)
        conv10 = Conv1D(128, kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)

        up11 = Conv1D(64,kernel_size=3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = up11)(conv10))
        conv11 = Conv1D(64, kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up11)
        conv11 = Conv1D(64, kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)

        up12 = Conv1D(32,kernel_size=3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = up12)(conv11))
        conv12 = Conv1D(32,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up12)
        conv12 = Conv1D(32,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv12)

        up13 = Conv1D(16,kernel_size=3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = up13)(conv12))
        conv13 = Conv1D(16,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up13)
        conv13 = Conv1D(16,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv13)

        up14 = Conv1D(8,kernel_size=3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = up14)(conv13))
        conv14 = Conv1D(8,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up14)
        conv14 = Conv1D(8,kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv14)

        up15 = Conv1D(4,kernel_size=3,activation = 'relu',padding = 'same',kernel_initializer = 'he_normal')(UpSampling1D(size = up15)(conv14))
        conv15 = Conv1D(4,kernel_size=3,activation = 'relu',padding = 'same',kernel_initializer = 'he_normal')(up15)
        conv15 = Conv1D(4,kernel_size=3, activation = 'relu', padding='same',  kernel_initializer = 'he_normal')(conv15)

        conv16 = Conv1D(1,kernel_size=1,activation = 'sigmoid',padding='same', kernel_initializer = 'he_normal')(conv15)

        model = Model(input = inputs, output = conv16)
        model.summary()

        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        model.compile(optimizer = 'sgd', loss = losses.mean_squared_error)
        return model

if __name__ == '__main__':
	output_path = r'./test'
	mkdir(output_path)
	filepath=[r'../data/train']
	trainnum,testnum = 750, 250
	img_rows = 73      # the number of stations
	img_cols = 4992    # the length of samples of one trace
	component = 1     # the components of the data
	save_img_path = output_path + r'/train_curves.png'
	log_path = output_path + r'/Log.txt'
    # ---------------
	batch_size = 32
	nb_epoch = 500
    #----------------
    # 2 Load data
	train_data, train_label, test_data, test_label = preprocessdata(filepath,trainnum,testnum,img_cols,component)
    # 3 Load model
    #myunet = myUnet(img_cols = img_cols,component = component)
    #model = myunet.get_une()
	myfcn = myFcn(img_cols = img_cols,component = component)
	model = myfcn.get_fcn()
    # 4 Train model
	print('Fitting model...')
	model_checkpoint = ModelCheckpoint(output_path+'/unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
	history = LossHistory()
	try:
		hist=model.fit(train_data, train_label, batch_size = batch_size, epochs = nb_epoch, verbose=2, validation_data=(test_data, test_label), callbacks=[model_checkpoint,history])
	finally:
        # 5 Print the training result
		history.loss_plot('epoch',save_img_path)
        # 6 save the result
		score = model.evaluate(test_data, test_label, verbose=1)
		history.write_log(log_path,score)
		model.save(output_path+'/test.h5')
