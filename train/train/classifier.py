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
        self.trainimages = glob.glob(self.train_path[0]+'*.jpg')
        self.trainimages1 = glob.glob(self.train_path[1]+'*.jpg')
        self.trainimages.extend(self.trainimages1)
        self.imagestrain = np.zeros([len(self.trainimages),self.height,self.width])
        self.trainlabels = None
        self.model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32, 32)),
        keras.layers.Dense(128, activation=tf.nn.relu),
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
        
    
    def create_labels(self):
        col_dir = self.train_path[0]+'*.jpg'
        col = glob.glob(col_dir)
        
        f = open('train_labels.txt','wb')
        for i in range(len(col)):
            f.write('0 ')
        
        col_dir = self.train_path[1]+'*.jpg'
        col = glob.glob(col_dir)
        for i in range(len(col)):
            f.write('1 ')
                        
    def scaletrain(self):
        self.imagestrain = self.imagestrain/255.0
          
    def preprocesstraining(self):
        i = 0
        for fname in self.trainimages:
            current = cv2.imread(fname)
            gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            res = cv2.resize(gray, dsize=(32,32), interpolation=cv2.INTER_CUBIC)
            self.imagestrain[i] = res
            i = i+1
        self.scaletrain()
              
    def train(self,epochs=500):
        self.model.fit(self.imagestrain, self.trainlabels, epochs=epochs, callbacks=[self.tensorboard])
    
    def save(self):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
# serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")
        
        
                            
                
    