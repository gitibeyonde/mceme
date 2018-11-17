import glob
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import dtype
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard
from time import time
from tensorflow.contrib.metrics.python.ops.confusion_matrix_ops import confusion_matrix
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
class classifier:
    
    def __init__(self,categories,train_path,height,width):
        #intialise training images
        init_op = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init_op)
        self.train_path = train_path
        self.height = height
        self.width = width
        self.trainimages = list()
        for i in train_path:
            print (i)
            self.trainimages.extend(glob.glob(i+'*.jpg'))
        print (len(self.trainimages))    
            
        self.imagestrain = np.zeros([len(self.trainimages),self.height,self.width])
        self.trainlabels = None
        self.model = keras.Sequential([
        keras.layers.Flatten(input_shape=(height, width)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
        self.model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  
        #self.model = tf.initialize_all_variables()
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time())) 
            
    def intialiselables(self):
        f = open('train_labels.txt')
        self.trainlabels = np.array([int(j) for j in f.read().split(' ')[:-1]])
        print (self.trainlabels.shape)
    
    def create_labels(self):
        label = 0
        count = 0
        f = open('train_labels.txt','wb')
        for j in self.train_path:
            if count>4:
                label = 1
            col_dir = j+'*.jpg'
            col = glob.glob(col_dir)
            for i in range(len(col)):
                f.write(str(label)+" ")
            count = count+1
                            
    def scaletrain(self):
        self.imagestrain = self.imagestrain/255.0
          
    def mixedprep(self,secondry):
        i = 0
        for fname in self.trainimages:
            second = cv2.imread(secondry[i])
            current = cv2.imread(fname)
            gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
            res1 = cv2.resize(gray, dsize=(self.height,self.width), interpolation=cv2.INTER_CUBIC)
            res2 = cv2.resize(gray1, dsize=(self.height,self.width), interpolation=cv2.INTER_CUBIC)
            res = np.concatenate((res1,res2),axis=1)
            self.imagestrain[i] = res
            i = i+1
        self.scaletrain() 
              
          
    def preprocesstraining(self):
        i = 0
        for fname in self.trainimages:
            current = cv2.imread(fname)
            gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            res = cv2.resize(gray, dsize=(self.height,self.width), interpolation=cv2.INTER_CUBIC)
            self.imagestrain[i] = res
            i = i+1
        print (len(self.trainimages))   
        self.scaletrain()
              
    def train(self,epochs=500):
        self.model.fit(self.imagestrain, self.trainlabels, epochs=epochs, callbacks=[self.tensorboard])
    
    def save(self):
        model_json = self.model.to_json()
        with open("model3.json", "w") as json_file:
            json_file.write(model_json)
# serialize weights to HDF5
        self.model.save_weights("model3.h5")
        print("Saved model to disk")
        
        
                            
                
    