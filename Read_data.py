import numpy as np
import matplotlib.pyplot as plt
from lab3_tools import *
'''
from lab1 import *
from lab2 import *
from prondict_ import *
'''
from IPython.display import Audio
import scipy.io.wavfile as wavfile
import os

# Bibliotek för att köra kod är lab3_tools, os och numpy

def delay(sound, fs, echo=0.07, amp=0.2, rep=7):
    #sound = sound.astype(np.float) / 2**15
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

    return delayed_sig




#phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
#phones = sorted(phoneHMMs.keys())
#nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
#stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

#lmfcc = concatenate([samples], True, False)


directory = 'tidigits/train/man/nw'
data = []
babble, _ = loadAudio("sorl2.wav")

# Hämtar en lista över alla filnamn i directory, loopar igenom den och skapar kopior med rev / noise
for filename in os.listdir(directory):

    gender = directory.split("/")[-2]       # Extraherar man/woman från path
    speaker = directory[-2:]                # Extraherar speaker-koden (två bokstäver) från path 
    f = os.path.join(directory, filename)   # Skapar lokal path till varje fil, t.ex. tidigits/train/man/nw/9a.wav

    if os.path.isfile(f):
        sample, samplerate = loadAudio(f)
        utter = filename.strip(".wav")      # Tar fram filnamn utan filformat, t.ex. 98z1.wav --> 98z1
        rev_samp = delay(sample, samplerate)
        noise_samp = sample + np.random.normal(0, 30, len(sample))
        bab_samp = sample + babble[:len(sample)]
        
        '''
        # En enkel plot över de fyra versionerna av ljudfilen
        fig, axs = plt.subplots(4)
        fig.subplots_adjust(hspace=2)
        axs[0].plot(sample) 
        axs[0].set_title("normal")
        axs[1].plot(rev_samp)
        axs[1].set_title("reverb")
        axs[2].plot(noise_samp)
        axs[2].set_title("noise")
        axs[3].plot(bab_samp)
        axs[3].set_title("babble")
        plt.show()
        break
        '''

        # Skapar lista med originalljud, reverbat ljud, ljud med noise och ljud med babbel
        data += [[sample, samplerate, "{} {} normal {}".format(gender, speaker, utter)], 
                [rev_samp, samplerate, "{} {} reverb {}".format(gender, speaker, utter)],
                [noise_samp, samplerate, "{} {} noise {}".format(gender, speaker, utter)],
                [bab_samp, smplrate, "{} {} babble {}".format(gender, speaker, utter)]]

