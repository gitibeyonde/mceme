from keras.models import model_from_json
import numpy as np
import tensorflow as tf
import cv2
import glob

class prediction:
    def __init__(self,model_path,model_weights,height,width,test_path=None):
        init_op = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init_op)
        if test_path!=None:
            self.test_path = test_path
            self.testlabels = None
            self.testimages = glob.glob(self.test_path[0]+'*.jpg')
            self.testimages1 = glob.glob(self.test_path[1]+'*.jpg')
            self.testimages.extend(self.testimages1)
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
        print("Loaded model from disk")
     
    def intialiselabels(self):
        f = open('test_labels.txt')
        self.testlabels = np.array([int(j) for j in f.read().split(' ')[:-1]]) 
     
    def scaletest(self):
        self.imagestest = self.imagestest/255.0 
    
    def preprocesstesting(self):
        i = 0 
        for fname in self.testimages:
            current = cv2.imread(fname)
            gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            res = cv2.resize(gray, dsize=(32,32), interpolation=cv2.INTER_CUBIC)
            self.imagestest[i] = res
            i = i+1        
        self.scaletest() 
      
    def create_labels(self):
        col_dir = self.test_path[0]+'*.jpg'
        col = glob.glob(col_dir)
        f = open('test_labels.txt','wb')
        for i in range(len(col)):
            f.write('0 ')
        
        col_dir = self.test_path[1]+'*.jpg'
        col = glob.glob(col_dir)
        for i in range(len(col)):
            f.write('1 ')
        
    def predict(self,image=None,height=32,width=32):
        if not image.any():
            predictions = self.model.predict(self.imagestest)
            new_predictions = [np.argmax(i) for i in predictions]
            return new_predictions
        else:
            imagestest = np.zeros([1,height,width])
            imagestest[0] = image
            predictions = self.model.predict(imagestest)    
            return predictions
    
    def printloss(self):
        test_loss, test_acc = self.model.evaluate(self.imagestest, self.testlabels)
        print test_loss,test_acc
    
    def printconfusionmatrix(self):
        new_predictions = self.predict()
        confusion_matrix = tf.confusion_matrix(self.testlabels,new_predictions)
        print (self.sess.run(confusion_matrix))      
    
    def getdifferentimages(self):
        new_predictions = self.predict()
        d = np.argwhere(self.test_labels!=new_predictions)
        for i in d:
            print self.testimages[i[0]]
            img = cv2.imread(self.testimages[i[0]])
            cv2.imshow('frame',img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                continue   
                
        
        
    
                     