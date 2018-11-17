import glob
'''
Created on 17-Oct-2018

@author: techsid
'''
from train import classifier
from train import preprocess
from prediction.predict import prediction

if __name__ == "__main__"  :
  
   
    new_classifier = classifier.classifier(categories=['cm75','cm100','cm125','cm150','cm175','gpm100','gpm150','gpm200','gpm250','gpm300','gpm350','gpm400'],train_path=['/home/admin/ai/data/new data22/crawling_man/100m/','/home/admin/ai/data/new data22/gp of men/100m/','/home/admin/ai/data/new data22/gp of men/150m/','/home/admin/ai/data/new data22/gp of men/200m/','/home/admin/ai/data/new data22/gp of men/250m/','/home/admin/ai/data/new data22/gp of men/300m/','/home/admin/ai/data/new data22/Hy Veh/150m/','/home/admin/ai/data/new data22/Hy Veh/200m/','/home/admin/ai/data/new data22/Hy Veh/250m/','/home/admin/ai/data/new data22/Hy Veh/300m/','/home/admin/ai/data/new data22/lt Veh/100m/','/home/admin/ai/data/new data22/lt Veh/200m/','/home/admin/ai/data/new data22/lt Veh/300m/']
                                          ,height=32,width=32)
    
    #new_1 = glob.glob('/Users/techsid/Documents/mixedaudiolift/*.jpg')
    #new_2 = glob.glob('/Users/techsid/Documents/mixedaudionolift/*.jpg')
    #new_1.extend(new_2)
    new_classifier.create_labels()
    new_classifier.intialiselables()
    new_classifier.preprocesstraining()
    #new_classifier.mixedprep(new_1)
    
    new_classifier.train(150)
    new_classifier.save()
    
    
    new_predict = prediction(model_path='model3.json',model_weights='model3.h5',test_path=['/Users/techsid/Desktop/ddctest/dur_1/ddctest/crawling_man/crawling_man_75m/','/Users/techsid/Desktop/ddctest/dur_1/ddctest/crawling_man/crawling_man_100m/','/Users/techsid/Desktop/ddctest/dur_1/ddctest/crawling_man/crawling_man_150m/','/Users/techsid/Desktop/ddctest/dur_1/ddctest/crawling_man/crawling_man_175m/','/Users/techsid/Desktop/ddctest/dur_1/ddctest/gp_of_man/gp_of_men_100m/','/Users/techsid/Desktop/ddctest/dur_1/ddctest/gp_of_man/gp_of_men_150m/','/Users/techsid/Desktop/ddctest/dur_1/ddctest/gp_of_man/gp_of_men_200m/','/Users/techsid/Desktop/ddctest/dur_1/ddctest/gp_of_man/gp_of_men_250m/','/Users/techsid/Desktop/ddctest/dur_1/ddctest/gp_of_man/gp_of_men_350m/']
                                          ,height=32,width=32)
    #new_1 = glob.glob('/Users/techsid/Documents/anL/*.jpg')
   # new_2 = glob.glob('/Users/techsid/Documents/aL/*.jpg')
   # new_1.extend(new_2)
    new_predict.create_labels()
    new_predict.intialiselabels()
    new_predict.preprocesstesting()
    #new_predict.mixedprep(new_1)
    new_predict.printloss()
    new_predict.printconfusionmatrix()
    
    