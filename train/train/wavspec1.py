'''
Created on 01-Nov-2018

@author: techsid
'''
from train import classifier
from train import preprocess
from prediction.predict import prediction  
  
preprocess.wav_to_spec('/Users/techsid/Desktop/eded/dur_1/eded/crawling_man/crawling_man_75m/')
preprocess.wav_to_spec('/Users/techsid/Desktop/eded/dur_1/eded/crawling_man/crawling_man_100m/')
preprocess.wav_to_spec('/Users/techsid/Desktop/eded/dur_1/eded/crawling_man/crawling_man_125m/')
preprocess.wav_to_spec('/Users/techsid/Desktop/eded/dur_1/eded/crawling_man/crawling_man_150m/')
preprocess.wav_to_spec('/Users/techsid/Desktop/eded/dur_1/eded/crawling_man/crawling_man_175m/')
preprocess.wav_to_spec('/Users/techsid/Desktop/eded/dur_1/eded/gp_of_man/gp_of_men_100m/')
preprocess.wav_to_spec('/Users/techsid/Desktop/eded/dur_1/eded/gp_of_man/gp_of_men_150m/')
preprocess.wav_to_spec('/Users/techsid/Desktop/eded/dur_1/eded/gp_of_man/gp_of_men_200m/')

