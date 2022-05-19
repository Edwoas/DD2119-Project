import numpy as np
from lab3_tools import *
from lab1_lmfcc import extract_lmfcc
#from lab2 import *
#from prondict_ import *
from IPython.display import Audio
import scipy.io.wavfile as wavfile
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt


# Bibliotek för att köra kod är lab3_tools, os och numpy
# Koden tar lång tid att köra, så kör prints i innersta for-loopen i read_data_test om ni vill experimentera

def loadAudio(filename):

    # float32 used for librosa's lmfcc method, originally int16
    samples, sampleRate = sf.read(filename, dtype='float32')

    return samples, sampleRate


def delay(sound, fs, echo=0.1, amp=0.2, rep=7):
    # sound = sound.astype(np.float) / 2**15
    delay = round(echo * fs)
    delayed_sig = np.concatenate((sound, np.zeros((delay * rep))))
    pad = np.copy(delayed_sig)
    decay = 0.6

    for i in range(rep):
        d = np.roll(pad, delay) * amp
        delayed_sig += d
        delay += round(delay * decay)
        decay -= 0.1
        amp /= 2

    # Removing trailing zeros 
    for n in reversed(range(0,len(delayed_sig))):
        if delayed_sig[n] == 0:
            delayed_sig = np.delete(delayed_sig, -1)
          
    return delayed_sig

def plot_mfcc(mfcc):    # plottar MFCC och filterbank
    '''
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].pcolormesh(filterbank[:,:, 0])
    ax[0].set_title('mspec')
    ax[1].pcolormesh(mfcc[:, :, 0])
    ax[1].set_title('mfcc')
    '''
    plt.pcolormesh(mfcc.T)

    plt.show()

def create_augmented_training_data():

    #path = 'tidigits/disc_4.1.1/tidigits/train'
    path = "tidigits/train"

    train_data = []
    babble, _ = loadAudio("sorl2.wav")

    # Hämtar en lista över alla filnamn i directory, loopar igenom den och skapar kopior med rev / noise
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                filename = os.path.join(root, file)
                samples, samplingrate = loadAudio(filename)
                utter = filename.strip(".wav")  # Denna tar även bort bokstaven för repetitionen??
                data_augments = [delay(samples, samplingrate), samples + np.random.normal(0, 30, len(samples)),
                                 samples + babble[:len(samples)], samples]  # gör data augmentation för varje sample
            
                aug_list = ["rev", "noise", "babble", "normal"]
                for i, u in zip(data_augments, aug_list): # för varje sample gör den nedanstående
                    mfcc = extract_lmfcc(i, samplingrate) # Denna tar ut MFCCs för alla samples, gör lab 1 i princip.
                    
                    train_data.append({'filename': filename, 'utter': utter, 'aug_type': u, 'mfcc': mfcc,
                                       'samplingrate': samplingrate})   # gör en dictionary med filnamn, utterance,
                                                                        # vilken typ av data augmentation, mfcc och 
                                                                        # samplingrate
    return train_data


""" 
# Extracting MFCC using Librosa, *currently not used*
def extract_mfcc(sample, samplerate):
    # Documentation: https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
    # Extracting mfcc as from lecture ?

    enframe = librosa.util.frame(sample, frame_length=400, hop_length=200) 
    preemph = librosa.effects.preemphasis(enframe, coef=0.97)  # kolla in coef = ?
    fft = librosa.stft(preemph, n_fft=512, win_length=400, hop_length=200)
    powerspectrum = np.abs(fft)**2
    log_powerspectrum = librosa.power_to_db(powerspectrum)
    filterbank = librosa.feature.melspectrogram(S=log_powerspectrum, sr=samplerate, n_fft=512)
    mfcc = librosa.feature.mfcc(S=filterbank, n_mfcc=13, n_fft=512) 
    return mfcc, filterbank """

x = create_augmented_training_data() # läser in datan

# TODO: 
# Skapa validation set, 10/90 