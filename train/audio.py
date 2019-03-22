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


CLIP_TIME = 2
CHUNK = 4096
CAT = ['GP OF MEN','VEHICLE','VEHICLE']
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = CLIP_TIME
WAVE_OUTPUT_FILENAME = "output.wav"
new_predict = prediction(model_path='../models/architecture/model12.json',model_weights='../models/weights/model12.h5'
		                                      ,height=32,width=32)
show_result = False
curr_result = ''
check_time = 1
result_count = 0
max_count = 1
gp = 'men_alert'
crawl = 'vehicle_alert'
vehiclecount = 0

def ongp():
	print ("ongpcalled")
	global CAT
	global gp,crawl
	CAT = ['GP OF MEN','GP OF MEN','GP OF MEN']
	gp = 'men_alert'
	crawl = 'men_alert'

def draw_men():
    imgpng = Image.open("men.png")
    ImageNumpyFormat =  np.asarray(imgpng)
    plt.imshow(ImageNumpyFormat)
    plt.draw()
    plt.pause(CLIP_TIME*2) # pause how many seconds
    plt.close()
 
def draw_vehicle():
    imgpng = Image.open("vehicle.png")
    ImageNumpyFormat =  np.asarray(imgpng)
    plt.imshow(ImageNumpyFormat)
    plt.draw()
    plt.pause(CLIP_TIME*2) # pause how many seconds
    plt.close()    

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
		

<<<<<<< Updated upstream
		call(["sox",WAVE_OUTPUT_FILENAME,"noise_rem.wav","noisered","live/noise2.prof","0.21"])
		call(["sox","noise_rem.wav","live/out.wav","silence","1","0.1","2%","-1","0.1","2%"])
=======
		call(["sox",WAVE_OUTPUT_FILENAME,"-n","noiseprof","noise.prof"])
		call(["sox",WAVE_OUTPUT_FILENAME,"noise_rem.wav","noisered","train/noise2.prof","0.21"])
		call(["sox","noise_rem.wav","ll/out.wav","silence","1","0.1","2%","-1","0.1","2%"])
>>>>>>> Stashed changes


		with contextlib.closing(wave.open('live/out.wav','r')) as f:
		    frames = f.getnframes()
		    rate = f.getframerate()
		    duration = frames / float(rate)
		    print (duration)
		    if duration<check_time:
    			curr_result = ''
    			result_count = 0
    			continue
<<<<<<< Updated upstream
		call(["sox",'live/out.wav','live/trimout.wav',"trim",'0',str(check_time)])
		check = preprocess.graphspectogram('live/trimout.wav')	
=======
		call(["sox",'ll/out.wav','ll/out1.wav',"trim",'0',str(check_time)])
		check = preprocess.graphspectogram('ll/out1.wav')	
>>>>>>> Stashed changes
		if check==0:
			print ("-----------------------silence or noise------------------------------------------")
			curr_result = ''
			result_count = 0
			continue
		#new_1 = glob.glob('/Users/techsid/Documents/anL/*.jpg')
		# new_2 = glob.glob('/Users/techsid/Documents/aL/*.jpg')
		# new_1.extend(new_2)

<<<<<<< Updated upstream
		current = cv2.imread('live/trimout.wav.jpg')
=======
		current = cv2.imread('ll/out1.wav.jpg')
>>>>>>> Stashed changes
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
				vehiclecount = 0
<<<<<<< Updated upstream
				subprocess.Popen([sys.executable,"..\\ai\\popup\\"+gp+".py"])
			else:
				vehiclecount = vehiclecount + 1
				subprocess.Popen([sys.executable, "..\\ai\\popup\\"+crawl+".py"])
=======
				subprocess.Popen([sys.executable,"C:\\Users\\siddharth\\Documents\\ai\\popup\\"+gp+".py"])
			else:
				vehiclecount = vehiclecount + 1
				subprocess.Popen([sys.executable, "C:\\Users\\siddharth\\Documents\\ai\\popup\\"+crawl+".py"])
>>>>>>> Stashed changes
				if vehiclecount>4:
					#ongp()
					vehiclecount = 0
				

			
except KeyboardInterrupt:
	pass	

