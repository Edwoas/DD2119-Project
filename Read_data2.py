import numpy as np
from lab3_tools import *
from lab1 import *
from lab2 import *
from prondict_ import *
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


directory = 'tidigits/disc_4.1.1/tidigits/train/man/nw'
data = []
babble, _ = loadAudio("sorl2.wav")

for filename in os.listdir(directory):  # Stegar igenom mapp med data och skapar kopior på filer fast med rev / noise på
    spkr = directory.split("/")[-2]
    f = os.path.join(directory, filename)

    if os.path.isfile(f):
        sample, smplrate = loadAudio(f)
        utter = filename.strip(".wav")
        rev_samp = delay(sample, smplrate)
        noise_samp = sample + np.random.normal(0, 30, len(sample))
        bab_samp = sample + babble[:len(sample)]
        data += [[sample, smplrate, "{} {}".format(spkr, utter)], [rev_samp, smplrate, "{} rev {}".format(spkr, utter)],
                 [noise_samp, smplrate, "{} noise {}".format(spkr, utter)],
                 [bab_samp, smplrate, "{} babble {}".format(spkr, utter)]]

