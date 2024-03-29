#!/usr/bin/python


from subprocess import call
from train import preprocess
from prediction.predict import prediction
import cv2,os
import pyaudio
import wave
import threading
import sys
from PIL import Image	
import subprocess
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import contextlib


FILE_SEPARATOR='/'
#FILE_SEPARATOR = '\\'
CLIP_TIME = 2
CHUNK = 4096
CAT = ['GP OF MEN','VEHICLE','VEHICLE']
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = CLIP_TIME
WAVE_OUTPUT_FILENAME = "live"+FILE_SEPARATOR+"output.wav"

new_predict = prediction(model_path='train'+FILE_SEPARATOR+'model12.json',model_weights='train'+FILE_SEPARATOR+'model12.h5'
		                                      ,height=32,width=32)
show_result = False
curr_result = ''
check_time = 1
result_count = 0
max_count = 2
gp = 'men_alert'
vehicle = 'vehicle_alert'
vehiclecount = 0


try:
	while True:
	
		p = pyaudio.PyAudio()
		stream = p.open(format=FORMAT,
		                channels=CHANNELS,
		                rate=RATE,
		                input=True,
		                frames_per_buffer=CHUNK)

		print("* recording")

		frames = []

		for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		    data = stream.read(CHUNK)
		    frames.append(data)

		print("* done recording")

		stream.stop_stream()
		stream.close()
		p.terminate()

		wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
		wf.setnchannels(CHANNELS)
		wf.setsampwidth(p.get_sample_size(FORMAT))
		wf.setframerate(RATE)
		wf.writeframes(b''.join(frames))
		wf.close()
		

		call(["sox",WAVE_OUTPUT_FILENAME,"noise_rem.wav","noisered","live"+FILE_SEPARATOR+"noise2.prof","0.21"])
		call(["sox","noise_rem.wav","live"+FILE_SEPARATOR+"out.wav","silence","1","0.1","2%","-1","0.1","2%"])


		with contextlib.closing(wave.open('live'+FILE_SEPARATOR+'out.wav','r')) as f:
		    frames = f.getnframes()
		    rate = f.getframerate()
		    duration = frames / float(rate)
		    print (duration)
		    if duration<check_time:
    			curr_result = ''
    			result_count = 0
    			continue
		call(["sox",'live'+FILE_SEPARATOR+'out.wav','live'+FILE_SEPARATOR+'trimout.wav',"trim",'0',str(check_time)])
		check = preprocess.graphspectogram('live'+FILE_SEPARATOR+'trimout.wav')	
		if check==0:
			print ("-----------------------silence or noise------------------------------------------")
			curr_result = ''
			result_count = 0
			continue
		#new_1 = glob.glob('/Users/techsid/Documents/anL/*.jpg')
		# new_2 = glob.glob('/Users/techsid/Documents/aL/*.jpg')
		# new_1.extend(new_2)

		current = cv2.imread('live'+FILE_SEPARATOR+'trimout.wav.jpg')
		gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
		res = cv2.resize(gray, dsize=(32,32), interpolation=cv2.INTER_CUBIC)
		result = new_predict.prdictr(image = res)
		#result,uncertainity = new_predict.predict_with_uncertainity(res)

		print (CAT[result[0]])
		print (result_count)
		print (curr_result)
		if CAT[result[0]]==curr_result:
			result_count = result_count+1
			if result_count==max_count:
				show_result = True
		else:
			result_count = 0	
			show_result = False
			curr_result = CAT[result[0]]	

		if show_result ==True:
			
			print (CAT[result[0]])
			print ("*****************************************************************************************")
			print (CAT[result[0]])
			print ("*****************************************************************************************")
			show_result ==False
			result_count = 0
			curr_result = ''

			if result[0]==0 :
			
				subprocess.Popen([sys.executable,"popup"+FILE_SEPARATOR+gp+".py"])
			else:
			
				subprocess.Popen([sys.executable, "popup"+FILE_SEPARATOR+vehicle+".py"])
				

			
except KeyboardInterrupt:
	pass	

