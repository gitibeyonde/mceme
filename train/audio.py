from subprocess import call
from train import preprocess
from prediction.predict import prediction
import cv2
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


CLIP_TIME = 3
CHUNK = 4096
CAT = ['GP OF MEN','GP OF MEN','VEHICLE','VEHICLE']
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = CLIP_TIME
WAVE_OUTPUT_FILENAME = "output.wav"
new_predict = prediction(model_path='..\\models\\architecture\\model7.json',model_weights='..\\models\\weights\\model7.h5'
		                                      ,height=32,width=32)
show_result = False
curr_result = ''
check_time = 1.8

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
		

		call(["sox",WAVE_OUTPUT_FILENAME,"-n","noiseprof","noise.prof"])
		call(["sox",WAVE_OUTPUT_FILENAME,"noise_rem.wav","noisered","noise.prof","0.21"])
		call(["sox","noise_rem.wav","ll/out.wav","silence","1","0.1","1%","-1","0.1","1%"])


		with contextlib.closing(wave.open('ll/out.wav','r')) as f:
		    frames = f.getnframes()
		    rate = f.getframerate()
		    duration = frames / float(rate)
		    print (duration)
		    if duration<check_time:
    			curr_result = ''
    			continue

		check =preprocess.wav_to_spec('ll/')	
		if check==0:
			print ("-----------------------silence or noise------------------------------------------")
			curr_result = ''
			continue
		#new_1 = glob.glob('/Users/techsid/Documents/anL/*.jpg')
		# new_2 = glob.glob('/Users/techsid/Documents/aL/*.jpg')
		# new_1.extend(new_2)

		current = cv2.imread('ll/out.wav.jpg')
		gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
		res = cv2.resize(gray, dsize=(32,32), interpolation=cv2.INTER_CUBIC)
		result = new_predict.prdictr(image = res)
		#result,uncertainity = new_predict.predict_with_uncertainity(res)

		

		if CAT[result[0]]==curr_result:
			show_result = True
		else:	
			show_result = False
			curr_result = CAT[result[0]]	

		if show_result ==True:
			
			print ("*****************************************************************************************")
			print (CAT[result[0]])
			print ("*****************************************************************************************")
			show_result ==False

			if result[0]==0 or result[0]==1:
				subprocess.Popen([sys.executable,"C:\\Users\\siddharth\\Documents\\ai\\popup\\men_alert.py"])
			else:
				subprocess.Popen([sys.executable, "C:\\Users\\siddharth\\Documents\\ai\\popup\\vehicle_alert.py"])
			
except KeyboardInterrupt:
	pass	