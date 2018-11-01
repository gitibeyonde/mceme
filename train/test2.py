from train import preprocess
from prediction import predict
import numpy as np
import glob

def twicepredict(image,audio):
    preprocess.graphspectogram(audio)
    imagespec = audio+'.jpg'
    audioimage = preprocess.imagepreprocess(imagespec, 32, 32)
    image = preprocess.imagepreprocess(image, 32, 32)
    model1 = predict.prediction('model.json','model.h5',height=32,width=32)
    model2 = predict.prediction('model1.json','model1.h5',height=32,width=32)
    result1 = model1.predict(audioimage)
    result2 = model2.predict(image)
    return result1,result2

def twicevaluepredict(path):
    
    model1 = predict.prediction('model.json','model.h5',height=32,width=32)
    model2 = predict.prediction('model1.json','model1.h5',height=32,width=32)
    images1 = glob.glob(path[0]) 
    images2 = glob.glob(path[1])
    result = list()
    lift,nolift = 0,0
    for i in range(len(images1)):
        audio = preprocess.imagepreprocess(images1[i], 32, 32)
        image = preprocess.imagepreprocess(images2[i], 32, 32)
        result1 = model1.predict(audio)
        result2 = model2.predict(image)
        result = result1*result2
        new_predictions = np.argmax(result)
        if new_predictions==0:
            lift = lift+1
        else:
            nolift = nolift+1
    return lift,nolift            
        
if __name__=="__main__":
    cat = ['lift','nonlift']
    #result1,result2 = twicepredict(audio='/Users/techsid/Documents/test_audio_unhappy/3bc21161_nohash_1.wav',image='/Users/techsid/Documents/test_nonlift/2012_004068.jpg')
    
    print twicevaluepredict(['/Users/techsid/Documents/aL/*.jpg','/Users/techsid/Documents/imnL/*.jpg'])
    