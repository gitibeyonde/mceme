import glob
'''
Created on 17-Oct-2018

@author: techsid
'''
from train import classifier
from train import preprocess
from prediction.predict import prediction

if __name__ == "__main__"  :
    #preprocess.wav_to_spec('/Users/techsid/Documents/test_audio_happy')
    new_classifier = classifier.classifier(categories=['happy','unhappy'],train_path=['/Users/techsid/Documents/mixedimagelift/','/Users/techsid/Documents/mixedimagenolift/']
                                          ,height=32,width=32)
    
    new_1 = glob.glob('/Users/techsid/Documents/mixedaudiolift/*.jpg')
    new_2 = glob.glob('/Users/techsid/Documents/mixedaudionolift/*.jpg')
    new_1.extend(new_2)
    new_classifier.create_labels()
    new_classifier.intialiselables()
    new_classifier.mixedprep(new_1)
    new_classifier.train(15)
    new_classifier.save()
    
    new_predict = prediction(model_path='model3.json',model_weights='model3.h5',test_path=['/Users/techsid/Documents/imL/','/Users/techsid/Documents/imnL/'],height=32,width=32)
    new_1 = glob.glob('/Users/techsid/Documents/anL/*.jpg')
    new_2 = glob.glob('/Users/techsid/Documents/aL/*.jpg')
    new_1.extend(new_2)
    new_predict.create_labels()
    new_predict.intialiselabels()
    new_predict.mixedprep(new_1)
   # new_predict.printloss()
    new_predict.printconfusionmatrix()
    