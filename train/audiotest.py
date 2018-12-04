from subprocess import call
from train import preprocess
from prediction.predict import prediction
import cv2,os
import glob
import time
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
from threading import Thread
import cmd


path = 'C:\\Users\\siddharth\\Documents\\ai\\doppler audio\\test_audio\\'
music_files = [path+'gp_of_men.wav',path+'vehicle.wav',path+'hy_veh.wav']
CLIP_TIME = 2.5
CHUNK = 4096
ACTUALCAT = {'GP':0,'LTVEH':1,'HVVEH':2}
CAT = ['GP','LTVEH','HVVEH']
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = CLIP_TIME
WAVE_OUTPUT_FILENAME = "ll/output"
new_predict = prediction(model_path='..\\models\\architecture\\model14.json',model_weights='..\\models\\weights\\model14.h5'
		     	                                 ,height=32,width=32)
show_result = False
curr_result = ''
check_time = 1
currentfile = -1 
RESULTS = [[0 for k in range(len(ACTUALCAT))] for i in range(len(ACTUALCAT))] 
POPUPS = 0
count = 0
silencecount = 0

def Thread1():
	try:
		global count,WAVE_OUTPUT_FILENAME
		while True:
			count = count+1
			count = count%10
			OUTPUT_FILENAME = WAVE_OUTPUT_FILENAME+str(count)+'.wav'
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
			wf = wave.open(OUTPUT_FILENAME, 'wb')
			wf.setnchannels(CHANNELS)
			wf.setsampwidth(p.get_sample_size(FORMAT))
			wf.setframerate(RATE)
			wf.writeframes(b''.join(frames))
			wf.close()
			
			#if duration==4.922630385487528:
			#	break
				
	
	except KeyboardInterrupt:
		pass

thread = Thread(target = Thread1, args = (),daemon = True)
thread.start()

try:
	for music_file in music_files:
			time.sleep(10)
			silencecount = 0
			currentfile = currentfile+1
			pro = subprocess.Popen(music_file,shell=True)
			print ("sleeping")	
			print ("music file started")
			while(silencecount<10):	
				for i in glob.glob('ll/*.wav'):
					print (i)
					call(["sox",i,"-n","noiseprof","noise.prof"])
					call(["sox",i,"noise_rem.wav","noisered","noise.prof","0.20"])
					call(["sox","noise_rem.wav",i,"silence","1","0.1","2%","-1","0.1","2%"])
					with contextlib.closing(wave.open(i,'r')) as f:
	
						frames = f.getnframes()
						rate = f.getframerate()
						duration = frames / float(rate)
						print (duration)
					if duration==0.0:
						silencecount = silencecount + 1
					else:
						silencecount = 0
					if duration<check_time:
						os.remove(i)
						continue
					call(["sox",i,'op.wav',"trim",'0',str(check_time)])				
					check = preprocess.graphspectogram('op.wav')	
					if check==0:
						
						print ("unable to create spectogram")
						continue
					#new_1 = glob.glob('/Users/techsid/Documents/anL/*.jpg')
					# new_2 = glob.glob('/Users/techsid/Documents/aL/*.jpg')
					# new_1.extend(new_2)
					
					os.remove(i)
					current = cv2.imread('op.wav'+'.jpg')
					gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
					res = cv2.resize(gray, dsize=(32,32), interpolation=cv2.INTER_CUBIC)
					result = new_predict.prdictr(image = res)
					#result,uncertainity = new_predict.predict_with_uncertainity(res)
	
					print ("*****************************************************************************************")
					print (CAT[result[0]])
					print ("*****************************************************************************************")
	
					POPUPS = POPUPS + 1
					RESULTS[currentfile][ACTUALCAT[CAT[result[0]]]] = RESULTS[currentfile][ACTUALCAT[CAT[result[0]]]] + 1
			
				time.sleep(10)

except KeyboardInterrupt:
	pass

print (RESULTS)			