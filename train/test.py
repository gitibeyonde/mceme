'''
Created on 17-Oct-2018

@author: techsid
'''
from train import classifier
from prediction import predict
from train import preprocess

if __name__ == "__main__"  :
    #preprocess.wav_to_spec('/Users/techsid/Documents/test_audio_happy')
    new_classifier = classifier.classifier(categories=['happy','unhappy'],train_path=['/Users/techsid/Documents/train_audio_happy/','/Users/techsid/Documents/train_audio_unhappy/']
                                          ,height=32,width=32)
    
    
    new_classifier.create_labels()
    new_classifier.intialiselables()
    new_classifier.preprocesstraining()
    new_classifier.train(500)
    new_classifier.save()
    new_predict = predict.prediction('model.json','model.h5',test_path=['/Users/techsid/Documents/test_audio_happy/','/Users/techsid/Documents/test_audio_unhappy/'],height=32,width=32)
    new_predict.create_labels()
    new_predict.intialiselabels()
    new_predict.preprocesstesting()
    new_predict.printconfusionmatrix()
    