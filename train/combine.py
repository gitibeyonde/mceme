import wave
import glob

path = 'F:\\oldcombinegpdata\\'
mylist = [path]
#infiles = glob.glob('F:\\new data22\\lt veh\\lt veh 200m\\*.wav')
#infiles = ["crawling man/crawling men 100m/crawling  men at 100m  m3.wav", "crawling man/crawling men 100m/crawling  men at 100m  m5.wav"]
newcou = 0
for i in mylist:
	print(i)
	outfile = path+"sounwds%d.wav"%newcou
	output = wave.open(outfile, 'wb')
	count = 0
	for infile in glob.glob(i+'\\*.wav'):
		print (infile)
		w = wave.open(infile, 'rb')
		if count == 0:
	    	 output.setparams(w.getparams())

		output.writeframes(w.readframes(w.getnframes()))
		w.close()
		count = count + 1  

	newcou = newcou+1
	output.close()
