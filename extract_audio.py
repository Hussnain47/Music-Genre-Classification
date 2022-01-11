import os
import librosa as lr
import tensorflow
from keras.models import model_from_json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SR = 22050

def extract_features(audio):
    features = {}
    
    features['length'] = lr.get_duration(audio,sr=SR)*SR
    ch = lr.feature.chroma_stft(audio,sr=SR)
    features['chroma_stft_mean'] = np.mean(ch)
    features['chroma_stft_var'] = np.var(ch)

    rms = lr.feature.rms(audio)
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)

    sc = lr.feature.spectral_centroid(audio,sr=SR)
    features['spectral_centroid_mean'] = np.mean(sc)
    features['spectral_centroid_var'] = np.var(sc)

    sb = lr.feature.spectral_bandwidth(audio,sr=SR)
    features['spectral_bandwidth_mean'] = np.mean(sb)
    features['spectral_bandwidth_var'] = np.var(sb)

    sro = lr.feature.spectral_rolloff(audio)
    features['rolloff_mean'] = np.mean(sro)
    features['rolloff_var'] = np.var(sro)

    scr = lr.feature.zero_crossing_rate(audio)
    features['zero_crossing_rate_mean'] = np.mean(scr)
    features['zero_crossing_rate_var'] = np.var(scr)

    hr = lr.effects.harmonic(audio)
    features['harmony_mean'] = np.mean(hr)
    features['harmony_var'] = np.var(hr)

    perc = lr.effects.percussive(audio)
    features['perceptr_mean'] = np.mean(perc)
    features['perceptr_var'] = np.var(perc)

    tm = lr.beat.tempo(audio)
    features['tempo'] = tm[0]
    return extract_mfcc(features,audio)

def extract_mfcc(features,audio):
      music_length = lr.get_duration(audio,sr=SR)  
      if music_length > 35:
          diff = 30 - music_length
          tstart = diff//2
          tend = 30 + diff//2
      elif music_length < 29:
          tstart = 0
          tend = music_length
      else:
          tstart = 0
          tend = music_length
      audio_cut = audio[round(tstart*SR,ndigits=None)
         :round(tend*SR, ndigits= None)]
      data = lr.feature.mfcc(audio_cut,n_mfcc=20)
      for i in range(0,len(data)):
          features['mfcc'+str(i+1)+'_mean'] = np.mean(data[i])
          features['mfcc'+str(i+1)+'_var'] = np.var(data[i])
      return features

def read_audio(path):
    audio,sr = lr.load(path,sr=SR)
    return extract_features(audio)

def getscaled_data(features):
    scaler = StandardScaler()
    df = pd.read_csv("./Data/features_30_sec.csv")
    scaler.fit(np.array(df.iloc[:,1:-1],dtype=float))
    features = np.array(features)
    features = features[np.newaxis,:]
    x = scaler.transform(np.array(features,dtype='float32'))
    return x
def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    loaded_model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy',metrics=['acc'])
    return loaded_model
def classify(features:dict):
    model = load_model()
    f = list(features.values())
    return np.argmax(model.predict(getscaled_data(f)))

def return_classified(path):
    res = classify(read_audio(path)) 
    if res == 0:
        return 'Blues'
    elif res == 1:
        return 'Classical'
    elif res == 2:
        return 'Country'
    elif res == 3:
        return 'Disco'
    elif res == 4:
        return 'Hiphop'
    elif res == 5:
        return 'Jazz'
    elif res == 6:
        return 'Metal'
    elif res == 7:
        return 'Pop'
    elif res == 8:
        return 'Reggae'
    elif res == 9:
        return 'Rock'
print(return_classified('./Data/genres_original/blues.00000.wav'))
