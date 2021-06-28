import pyaudio
import wave
import numpy as np
import pandas as pd
import pickle
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

# to do load model from pkl
with open("svm.pkl", 'rb') as fid:
    model = pickle.load(fid)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "file.wav"

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print("recording...")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("finished recording")

# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

(rate, sig) = wav.read("file.wav")
mfcc_feat = mfcc(sig, rate)
fbank_feat = logfbank(sig, rate)
mfcc_feat= np.mean(mfcc_feat, axis=0).reshape([1, 13])
fbank_feat= np.mean(fbank_feat, axis=0).reshape([1, 26])
#print(mfcc_feat[1])
#print(fbank_feat[1])
mfcc_names, fbank_names = [], []
for i in range(len(mfcc_feat[0])):
    mfcc_names.append('mfcc_'+str(i))
for i in range(len(fbank_feat[0])):
    fbank_names.append('fbank_'+str(i))

data1 = pd.concat([pd.DataFrame(mfcc_feat, columns=mfcc_names), pd.DataFrame(fbank_feat, columns=fbank_names)], axis=1)
pred = model.predict(data1)
print(pred[0])
