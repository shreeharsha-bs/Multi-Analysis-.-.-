#!/usr/bin/python
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import sys
import pywt;import math
#import matplotlib.pyplot as plt
import subprocess

def PadRight(arr):
    nextPower = NextPowerOfTwo(len(arr))
    deficit = int(math.pow(2, nextPower) - len(arr))
    arr = np.concatenate((arr,np.zeros(deficit, dtype=arr.dtype)))
    return arr

def NextPowerOfTwo(number):
    # Returns next power of two following 'number'
    return math.ceil(math.log(number,2))

subprocess.check_call("/home/swar/swar/shreeharsha/best_VAD/VAD/twoPass.sh '%s'" % sys.argv[1], shell=True)

def normalize(x):
	return x/(0.0+np.max(np.abs(x)))

wav_name = sys.argv[1]
fs,aud = wav.read(wav_name)
old_length = len(aud)
aud = PadRight(aud)
aud = normalize(aud)
vad2 = np.array([line.rstrip('\n') for line in open(wav_name+'_twopass.txt_pp.txt')]).astype(float)
vad = np.repeat(vad2,len(aud)//len(vad2))
sil = aud[np.where(aud[:len(vad)]*(1-vad)!=0)[0]]
#plt.plot(np.fft.fft(sil));plt.show()

l1=int(sys.argv[2])
l2=l1/2
sil2 = np.zeros(l1)
#N = 21
#sil2 = savgol_filter(sil,N,9) # Nope lol
for i in range(0,len(sil)-l1,l1):
	sil2+=sil[i:i+l1]

#plt.plot(sil2);plt.plot(sil);plt.show()
sil_spec = np.abs(np.fft.rfft(sil2,l1-1))
#wav_spec = np.sqrt(1-np.square(np.abs(sil_spec)))
#pdb.set_trace()
for i in range(1,int(np.log2(l1))):
	x = 2**(-i)*np.abs(np.fft.rfft(sil2,(l1-1)//2**i))
	sil_spec = sil_spec*(np.repeat(x,len(sil_spec)//len(x)))

#scal_func = sil_spec/(0.0+np.max(np.abs(sil_spec)))
#wav_func = np.sqrt(1-np.square(np.abs(scal_func))) #wav_spec/(0.0+np.max(np.abs(wav_spec)))

#wav = np.fft.rfft(scal_func)
#let = wav[::-1]
#wavelet = list(let)+list(wav[1:])
#wavelet = wavelet/(0.0+np.max(np.abs(wavelet)))
H0 = sil_spec
H0 = normalize(np.convolve(H0,np.ones(l1//100),'same'))
H0 = normalize(np.convolve(H0,np.ones(l1//100),'same'))
#H1 = list(H0)
#H1.reverse()
H1 = np.sqrt(1-np.square(np.abs((H0))))
#plt.plot(H0);plt.plot(H1);plt.show()
h0 = np.fft.irfft(H0)
h1 = h0*[(-1)**k for k in range(len(h0))]
h2 = normalize(np.fft.irfft(-np.flipud(H1)))
h3 = normalize(np.fft.irfft(np.flipud(H0)))
#plt.plot(h0);plt.plot(h1);plt.show()
#wav2 = np.fft.rfft(wav_func)
#let2 = wav2[::-1]
#wavelet2 = list(let2[0:-1])+list(wav2[0:])
#wavelet2 = wavelet2/(0.0+np.max(np.abs(wavelet2)))

#plt.plot(wavelet)
#plt.plot(scal_func)
#plt.plot(wav_func)

#plt.plot(np.abs(np.fft.fft(wavelet)))
#plt.plot(np.abs(np.fft.fft(wavelet2)))
#plt.show()
dec_lo, dec_hi, rec_lo, rec_hi = h0, h1, h2, h3
filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi]
myWavelet = pywt.Wavelet(name="custom_wav", filter_bank=filter_bank)


[(ca2,cd2),(ca1,cd1)] = pywt.swt(aud,myWavelet,level =2)
ca2=ca2[:old_length]
cd1=cd1[:old_length]
cd2=cd2[:old_length]
#ca = signal.convolve(aud,h0,mode='same')

#cd = signal.convolve(aud,h1,mode='same')
#ca2 = signal.convolve(ca,h0,mode='same')
#cd2 = signal.convolve(ca,h1,mode='same')

#CA = np.zeros(2*len(ca))
#CD = np.zeros(2*len(cd))
#CA[::2] = ca
#CD[::2] = cd # Use low pass filter after upsampling as well!!
#CA[1:-1:2] = (CA[0:-2:2]+CA[2:-1:2])/2
#CD[1:-1:2] = (CD[0:-2:2]+CD[2:-1:2])/2
#ca = signal.resample_poly(ca2,2,1)
#cd = signal.resample_poly(cd1,100,50)
#cd2 = signal.resample_poly(cd2,4,1)

#plt.plot(3+normalize(ca2))
#plt.plot(4+normalize(cd1))

wav.write(wav_name[0:-4]+'_app2.wav',fs,normalize(ca2))
app2 = wav_name[0:-4]+'_app2.wav'
subprocess.check_call("python 32bit216bitpcm.py '%s'" % app2, shell=True)
wav.write(wav_name[0:-4]+'_det2.wav',fs,normalize(cd2))
det2 = wav_name[0:-4]+'_det2.wav'
subprocess.check_call("python 32bit216bitpcm.py '%s'" % det2, shell=True)
wav.write(wav_name[0:-4]+'_det1.wav',fs,normalize(cd1))
det1 = wav_name[0:-4]+'_det1.wav'
subprocess.check_call("python 32bit216bitpcm.py '%s'" % det1, shell=True)
#wav.write(wav_name[0:-4]+'_app2.wav',fs,normalize(ca2))
#app2 = wav_name[0:-4]+'_app2.wav'
#subprocess.check_call("python 32bit216bitpcm.py '%s'" % app2, shell=True)
#wav.write(wav_name[0:-4]+'_det2.wav',fs,normalize(cd2))
#det2 = wav_name[0:-4]+'_det2.wav'
#subprocess.check_call("python 32bit216bitpcm.py '%s'" % det2, shell=True)


#plt.plot(aud)
#plt.show()


