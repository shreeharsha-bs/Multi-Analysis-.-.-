import scipy.io.wavfile as wav
import sys
import numpy as np

wav32 = sys.argv[1] #input wav file
fs, data = wav.read(wav32)
pcm16 = np.zeros(len(data),dtype = 'int16')
d1 = max(data)
for i in range(0,len(data)):
    pcm16[i] = int(data[i]*32767)

#pcm16 = np.astype(np.int16)
wav.write(wav32,fs,pcm16)
