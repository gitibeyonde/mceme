import glob
import uuid
from scipy.io import wavfile
from scipy.fftpack import fft
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import os,cv2
import numpy as np
import fnmatch
#read wave file
def readwave(file):
    sr,data = wavfile.read(file)
    return sr,data

def plot_images(folder):
    uuidu = str(uuid.uuid4().hex)
    os.chdir('../spectrumanalysis')
    os.mkdir(uuidu)
    saving_folder = uuidu+'/'
    i = 0
    plt.subplots_adjust(hspace=0.1)
    for root, dirnames, filenames in os.walk(folder):
        j = 0
        for filename in fnmatch.filter(filenames, '*.jpg'):
            if (i==8):
                j = j+1
                plt.savefig(saving_folder+root.split('/')[-1]+str(j)+'.png',
                    dpi=900)
                i=0
                plt.close()
            ax = plt.subplot(2,4,i+1)
            ax.set_title(filename,size=3)
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            im = cv2.imread(os.path.join(root, filename))
            i+=1
            imgplot = ax.imshow(im)
    return
    
    
def imagepreprocess(image,height,width):
    i = 0 
    current = cv2.imread(image)
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(gray, dsize=(height,width), interpolation=cv2.INTER_CUBIC)       
    res = res/255.0
    return res 

def fourierfft(wav_file):
    sr,data = readwave(wav_file)
    fft_out = fft(data)
    plt.plot(data, np.abs(fft_out))
    plt.savefig(wav_file.split('.wav')[0]+'.png',
                dpi=100,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)
    plt.close()
    try:
        im = Image.open(wav_file.split('.wav')[0] + '.png')
        rgb_im = im.convert('RGB')
        rgb_im.save(wav_file.split('.png')[0] + '.jpg')
    except Exception as e:
        print (e)
    if os.path.exists(wav_file.split('.wav')[0] + '.png'):
        #os.system('convert '+(wav_file.split('.wav')[0] + '.png') + ' '+(wav_file.split('.wav')[0] + '.jpg'))
        os.remove(wav_file.split('.wav')[0] + '.png')
      
 


def npfourierfft(wav_file):
    sr,data = readwave(wav_file)
    fft_out = fft(data)
    print (np.abs(fft_out).shape)
    print (fft_out.shape)  
      

def getspecdata(wav_file):

    sr,data = readwave(wav_file)
    if (len(data)==0):
        return 0
    print ("reading file "+wav_file+" with size %d"%(len(data)))
    nfft = 256
    sf = 256
    pxx, freqs, bins, im = plt.specgram(data, nfft, sf)
    print (pxx.shape)
    return pxx



def graphspectogram(wav_file):
    sr,data = readwave(wav_file)
    if (len(data)==0):
        return 0
    print ("reading file "+wav_file+" with size %d"%(len(data)))
    nfft = 256
    sf = 256
    pxx, freqs, bins, im = plt.specgram(data, nfft, sf)
#    print ("pxx : ", len(pxx))
 #   print ("freqs : ", len(freqs))
  #  print ("bins : ", len(bins))
    print (freqs)
    print (pxx.shape)
    print (freqs.shape)

    #plt.show()
    plt.savefig(wav_file.split('.wav')[0] + '.png',
                dpi=100,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png
    plt.close()
    try:
        im = Image.open(wav_file.split('.wav')[0] + '.png')
        rgb_im = im.convert('RGB')
        rgb_im.save(wav_file.split('.png')[0] + '.jpg')
    except Exception as e:
        print (e)
    if os.path.exists(wav_file.split('.wav')[0] + '.png'):
        #os.system('convert '+(wav_file.split('.wav')[0] + '.png') + ' '+(wav_file.split('.wav')[0] + '.jpg'))
        os.remove(wav_file.split('.wav')[0] + '.png')
    return 1
#iterate over all the folders and create the spectogram     

def wav_to_spec(path):
    files = glob.glob(path + "/*.wav")
    for f in files:
        try:
            #print ("generating spectogram")
            val = graphspectogram(f)
            return val
        except Exception as e:
            print ("Something went wrong while generating spectrogram:", e) 

            
def savespecdata(path):
    data = list()
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.wav'):
            try:
                print (os.path.join(root, filename))
                data.append(getspecdata(os.path.join(root, filename)))
            except Exception as e:
                print ("Something went wrong while generating spectrogram:", e) 
    print(data[0])            
    new = np.array(data)
    np.savetxt('train.txt',new,fmt ='%d')            
    

def fourier(path):
    files = glob.glob(path + "/*.wav")
    for f in files:
        try:
   #         print ("generating spectogram")
            fourierfft(f)
        except Exception as e:
            print ("Something went wrong while generating fourier:", e)            