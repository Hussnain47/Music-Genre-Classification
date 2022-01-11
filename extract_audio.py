import os
import librosa as lr
from numpy.lib.function_base import diff
from tensorflow import keras
from keras.models import model_from_json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

MEAN = np.array([ 6.62030846e+05,  3.78681698e-01,  8.63398230e-02,  1.30929704e-01,
        3.05139928e-03,  2.20178090e+03,  4.69691575e+05,  2.24254107e+03,
        1.37079155e+05,  4.57154930e+03,  1.84434504e+06,  1.03612290e-01,
        2.98615041e-03, -3.65933186e-04,  1.25399936e-02, -3.95331702e-04,
        5.67007299e-03,  1.19505363e+02, -1.44472987e+02,  3.74787541e+03,
        9.95542780e+01,  7.06899539e+02, -8.92029111e+00,  4.68286436e+02,
        3.62924470e+01,  2.20447219e+02, -1.14484017e+00,  1.74404462e+02,
        1.46334905e+01,  1.27232071e+02, -5.12903824e+00,  1.16367019e+02,
        1.01191719e+01,  8.81989877e+01, -6.99406180e+00,  8.85669334e+01,
        7.72978464e+00,  8.17334899e+01, -6.02106854e+00,  7.55512280e+01,
        4.47169324e+00,  6.86708173e+01, -4.79588400e+00,  6.78717120e+01,
        1.78190006e+00,  6.47640963e+01, -3.86930761e+00,  6.28737606e+01,
        1.14814398e+00,  6.07309578e+01, -3.96602826e+00,  6.26336244e+01,
        5.07696354e-01,  6.37125862e+01, -2.32876090e+00,  6.62319299e+01,
       -1.09534803e+00,  7.01260962e+01])
VAR = np.array([3.17973709e+06, 6.66901285e-03, 5.97750033e-05, 4.30995696e-03,
       1.31963789e-05, 5.12086981e+05, 1.60559703e+11, 2.76732021e+05,
       9.29439187e+09, 2.47748862e+06, 2.02883522e+12, 1.74723744e-03,
       9.13249186e-06, 2.83283709e-06, 1.35664048e-04, 1.16951775e-06,
       4.22119717e-05, 7.98064776e+02, 1.00358878e+04, 7.66831782e+06,
       9.80641983e+02, 1.92633772e+05, 4.70014955e+02, 8.23689163e+04,
       2.77433349e+02, 1.34975154e+04, 1.49211798e+02, 1.01440000e+04,
       1.40035749e+02, 4.68415249e+03, 9.87139339e+01, 3.40715508e+03,
       1.09370548e+02, 1.68327084e+03, 6.85547240e+01, 1.55822195e+03,
       6.29378701e+01, 1.34801163e+03, 4.64251953e+01, 1.45283787e+03,
       4.50566907e+01, 1.06336104e+03, 3.80330604e+01, 1.09810678e+03,
       2.50588350e+01, 1.18563565e+03, 2.37282137e+01, 1.14833859e+03,
       2.09457941e+01, 1.14007898e+03, 2.06790442e+01, 1.11973411e+03,
       1.49550055e+01, 1.18231255e+03, 1.40931025e+01, 1.38057126e+03,
       1.47078986e+01, 2.04357264e+03])

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

    sro = lr.feature.spectral_rolloff(audio,sr=SR)
    features['rolloff_mean'] = np.mean(sro)
    features['rolloff_var'] = np.var(sro)

    scr = lr.zero_crossings(audio, pad = False)
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
    x = scaler.transform(np.array(df.iloc[0:1,1:-1],dtype='float32'))
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
