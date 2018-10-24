import glob
from scipy.io import wavfile
import matplotlib.pyplot as plt
from PIL import Image
import os,cv2
#read wave file
def readwave(file):
    sr,data = wavfile.read(file)
    return sr,data

def imagepreprocess(image,height,width):
    i = 0 
    current = cv2.imread(image)
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(gray, dsize=(height,width), interpolation=cv2.INTER_CUBIC)       
    res = res/255.0
    return res 

def graphspectogram(wav_file):
    sr,data = readwave(wav_file)
    print "reading file "+wav_file+" with size %d"%(len(data))
    nfft = 256
    sf = 256
    pxx, freqs, bins, im = plt.specgram(data, nfft, sf)
    print "pxx : ", len(pxx)
    print "freqs : ", len(freqs)
    print "bins : ", len(bins)
    plt.savefig(wav_file.split('.wav')[0] + '.png',
                dpi=100,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png
    try:
        im = Image.open(wav_file.split('.wav')[0] + '.png')
        rgb_im = im.convert('RGB')
        rgb_im.save(wav_file.split('.png')[0] + '.jpg')
    except Exception as e:
        print e
    if os.path.exists(wav_file.split('.wav')[0] + '.png'):
        #os.system('convert '+(wav_file.split('.wav')[0] + '.png') + ' '+(wav_file.split('.wav')[0] + '.jpg'))
        os.remove(wav_file.split('.wav')[0] + '.png')

#iterate over all the folders and create the spectogram            
def wav_to_spec(path):
    files = glob.glob(path + "/*.wav")
    for f in files:
        try:
            print "generating spectogram"
            graphspectogram(f)
        except Exception as e:
            print "Something went wrong while generating spectrogram:", e  