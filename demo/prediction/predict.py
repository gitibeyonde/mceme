from keras.models import model_from_json
import numpy as np
import tensorflow as tf
import cv2
import glob
import keras.backend as K

class prediction:
    def __init__(self,model_path,model_weights,height,width,test_path=None):
        init_op = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init_op)
        self.height = height
        self.width = width
        if test_path!=None:
            self.test_path = test_path
            self.testlabels = None
            self.testimages = list()
            for i in test_path:
                print (i)
                self.testimages.extend(glob.glob(i+'*.jpg'))
            print (len(self.testimages))
            self.imagestest = np.zeros([len(self.testimages),height,width])
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(model_weights)
        self.model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        self.f = K.function([self.model.layers[0].input, K.learning_phase()],
               [self.model.layers[-1].output])

        print("Loaded model from disk")
     
    def intialiselabels(self):
        f = open('test_labels.txt')
        self.testlabels = np.array([int(j) for j in f.read().split(' ')[:-1]]) 
     
    def scaletest(self):
        self.imagestest = self.imagestest/255.0 
    
    def mixedprep(self,secondry):
        i = 0
        for fname in self.testimages:
            second = cv2.imread(secondry[i])
            current = cv2.imread(fname)
            gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
            res1 = cv2.resize(gray, dsize=(self.height,self.width), interpolation=cv2.INTER_CUBIC)
            res2 = cv2.resize(gray1, dsize=(self.height,self.width), interpolation=cv2.INTER_CUBIC)
            self.imagestest[i] = np.concatenate((res1,res2),axis=1)
            i = i+1
        self.scaletest() 
    
    def preprocesstesting(self):
        i = 0 
        for fname in self.testimages:
            current = cv2.imread(fname)
            gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            res = cv2.resize(gray, dsize=(self.height,self.width), interpolation=cv2.INTER_CUBIC)
            self.imagestest[i] = res
            i = i+1        
        self.scaletest() 
      
    
    def create_labels(self):
        label = 0
        f = open('test_labels.txt','wb')
        for j in self.test_path:
            col_dir = j+'*.jpg'
            col = glob.glob(col_dir)
            for i in range(len(col)):
                f.write(str(label)+" ")
            label = label+1


    def create_sep_labels(self):
        count = 0
        label = 0
        f = open('test_labels.txt','wb')
        for j in self.test_path:
            if count==1:
                label = 1
            elif count ==6:
                label = 2
            elif count ==6:
                label = 2
            elif count ==10:
                label = 3
            col_dir = j+'*.jpg'
            col = glob.glob(col_dir)
            for i in range(len(col)):
                f.write(str(label)+" ")
            count = count+1
    
    def prdictr(self,image=None,height=32,width=32):
        imagestest = np.zeros([1,height,width])
        imagestest[0] = image
        predictions = self.model.predict(imagestest)
        prediction = [np.argmax(i) for i in predictions]    
        return prediction
    
        
    def predict(self,image=None,height=32,width=32):
        
        predictions = self.model.predict(self.imagestest)
        new_predictions = [np.argmax(i) for i in predictions]
        return predictions
        
            
    def printloss(self):
        test_loss, test_acc = self.model.evaluate(self.imagestest, self.testlabels)
        print (test_loss,test_acc)
    
    def printconfusionmatrix(self):
        new_predictions = self.predict()
        confusion_matrix = tf.confusion_matrix(self.testlabels,new_predictions)
        print (self.sess.run(confusion_matrix))      
    
    def getdifferentimages(self):
        new_predictions = self.predict()
        d = np.argwhere(self.test_labels!=new_predictions)
        for i in d:
            print (self.testimages[i[0]])
            img = cv2.imread(self.testimages[i[0]])
            cv2.imshow('frame',img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                continue   