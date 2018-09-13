import os
from scipy.io import wavfile
from scipy import signal as scisig
import numpy as np
import matplotlib.pyplot as plt


def snr(signal, recon):
    """Returns signal-noise ratio in dB."""
    ratio = np.var(signal)/np.var(signal-recon)
    return 10*np.log10(ratio)


def dcGC(t, f):
    """Dynamic compressive gammachirp filter as defined by Irino,
    with parameters from Park as used in Charles, Kressner, & Rozell.
    The log term is regularized to log(t + 0.00001).
    t : time in seconds, greater than 0
    f : characteristic frequency in Hz
    One but not both arguments may be numpy arrays.
    """
    ERB = 0.1039*f + 24.7
    return t**3 * np.exp(-2*np.pi*1.14*ERB*t) * \
        np.cos(2*np.pi*f*t + 0.979*np.log(t+0.000001))


# adapted from scipy cookbook
lowcut = 100
highcut = 6000


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scisig.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y


class SignalSet:

    def __init__(self,
                 sample_rate=16000,
                 data='../Data/speech_corpora/TIMIT/',
                 min_length=800,
                 seg_length=80000):
        self.sample_rate = sample_rate
        self.min_length = min_length
        self.seg_length = seg_length
        if isinstance(data, str):
            self.load_from_folder(data)
        else:
            self.data = data
            self.ndata = len(data)

    def load_from_folder(self, folder="/home/edodds/Data/TIMIT/", cache=True):
        min_length = self.min_length
        files = os.listdir(folder)
        if 'cache.npy' in files:
            self.data = np.load(os.path.join(folder, 'cache.npy'))
            print("Loaded data from cache.npy")
            return
        self.data = []
        for ff in files:
            if ff.endswith('.wav'):
                file = os.path.join(folder, ff)
                rate, signal = wavfile.read(file)
                if rate != self.sample_rate:
                    raise NotImplementedError('The signal in ' + ff +
                                              ' does not match the given' +
                                              ' sample rate.')
                if signal.shape[0] > min_length:
                    # bandpass
                    signal = signal/signal.std()
                    signal = butter_bandpass_filter(signal, lowcut, highcut,
                                                    self.sample_rate, order=5)
                    self.data.append(signal)
        self.ndata = len(self.data)
        print("Found ", self.ndata, " files")
        if cache:
            np.save(os.path.join(folder, 'cache.npy'), self.data)

    def rand_stim(self):
        """Get one random signal."""
        not_found = True
        while not_found:
            which = np.random.randint(low=0, high=self.ndata)
            signal = self.data[which]
            excess = signal.shape[0] - self.seg_length
            if excess >= 0:
                where = np.random.randint(low=0, high=excess)
                segment = signal[where:where+self.seg_length]
                not_found = False
        segment /= np.max(np.abs(segment))  # norm by max as in Smith & Lewicki
        return segment

    def get_batch(self, batch_size):
        return [self.rand_stim() for _ in range(batch_size)]

    def write_sound(self, filename, signal):
        signal /= np.max(signal)
        wavfile.write(filename, self.sample_rate, signal)

    def tiled_plot(self, stims):
        """Tiled plots of the given signals. Zeroth index is which signal.
        Kind of slow, expect about 10s for 100 plots."""
        nstim = stims.shape[0]
        plotrows = int(np.sqrt(nstim))
        plotcols = int(np.ceil(nstim/plotrows))
        f, axes = plt.subplots(plotrows, plotcols, sharex=True, sharey=True)
        for ii in range(nstim):
            axes.flatten()[ii].plot(stims[ii])
        f.subplots_adjust(hspace=0, wspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.setp([a.get_yticklabels() for a in f.axes[:-1]], visible=False)
