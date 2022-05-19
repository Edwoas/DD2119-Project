import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.signal.windows import hamming
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct


# - - - - - -  - - - FUNCTIONS FROM LAB 1 – USED FOR COMPUTING LMFCC - - - - - - - - - - -

def lifter(mfcc, lifter=22):
    nframes, nceps = mfcc.shape
    cepwin = 1.0 + lifter/2.0 * np.sin(np.pi * np.arange(nceps) / lifter)
    return np.multiply(mfcc, np.tile(cepwin, nframes).reshape((nframes,nceps)))

def trfbank(fs, nfft, lowfreq=133.33, linsc=200/3., logsc=1.0711703, nlinfilt=13, nlogfilt=27, equalareas=False):
    nfilt = nlinfilt + nlogfilt
    freqs = np.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
    if equalareas:
        heights = np.ones(nfilt)
    else:
        heights = 2./(freqs[2:] - freqs[0:-2])

    fbank = np.zeros((nfilt, nfft))

    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank


def enframe(samples, winlen, winshift):
    signal_length = len(samples)

    N = int(np.ceil((signal_length - winlen + winshift)/winshift ))
    total_rows = N - 1
    i = 0
    frame_matrix = np.zeros((total_rows, winlen))
    for row in range(0, total_rows):   # Row indeces go from 0 to 91 (tot 92)
        for col in range(0, winlen):   # Col indeces go from 0 to 399 (tot 400)

            frame_matrix[row][col] = samples[i]
            i += 1
            if i == signal_length - winlen: # last window reached
                break
        i -= winshift

    return frame_matrix
    
def preemp(matrix, p=0.97):

    preemp_matrix = np.zeros(matrix.shape)

    a = [1]
    b = [1, -p]
    for i in range(len(matrix)):
        preemp_matrix[i] = lfilter(b, a, matrix[i])

    return preemp_matrix

def windowing(matrix):

    window = hamming(len(matrix[0]), sym=False)
    
    windowed_matrix = np.zeros(matrix.shape)

    for i in range(len(matrix)):
        windowed_matrix[i] = matrix[i] * window

    return windowed_matrix


def powerSpectrum(matrix, nfft):

    fft_matrix = np.zeros((len(matrix), nfft))

    for i in range(len(matrix)):
        fft_matrix[i] = abs(fft(matrix[i], n = nfft))**2

    return fft_matrix
    
    

def logMelSpectrum(matrix, samplingrate):

    nfft = len(matrix[0])
    fbank = trfbank(samplingrate, nfft)

    mel_matrix = np.log(matrix @ fbank.T)
    
    return mel_matrix


def cepstrum(matrix, nceps):
    dct_matrix = np.zeros((len(matrix), nceps))

    for i in range(len(matrix)):
        dct_temp = dct(matrix[i])
        dct_matrix[i] = dct_temp[:nceps] # selecting the n first coefficients from the dct results

    ceps_matrix = lifter(dct_matrix)

    return ceps_matrix


def extract_lmfcc(samples, sr):
  
    enframed = enframe(samples, 400, 200)
    
    preemped = preemp(enframed)

    windowed = windowing(preemped)

    transformed = powerSpectrum(windowed, 512)

    mel_speced = logMelSpectrum(transformed, sr)

    return cepstrum(mel_speced, 20) # 20 COEFFICIENTS 