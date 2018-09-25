import os
from scipy.io import wavfile
from scipy import signal as scisig
import numpy as np
import matplotlib.pyplot as plt


def snr(signal, recon):
    """Returns signal-noise ratio in dB."""
    ratio = np.var(signal)/np.var(signal-recon)
    return 10*np.log10(ratio)


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
                 data="/home/edodds/Data/TIMIT/",
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
            self.ndata = len(self.data)
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
        # segment /= np.max(np.abs(segment))  # norm by max as in Smith & Lewicki
        segment = (500*len(segment)/self.sample_rate) * segment / np.linalg.norm(segment) # norm 1 per 2ms
        return segment

    def get_batch(self, batch_size):
        return [self.rand_stim() for _ in range(batch_size)]

    def write_sound(self, filename, signal):
        signal /= np.max(signal)
        wavfile.write(filename, self.sample_rate, signal)

    def tiled_plot(self, stims, trim=False):
        """Tiled plots of the given signals. Zeroth index is which signal.
        Kind of slow, expect about 10s for 100 plots."""
        nstim = stims.shape[0]
        plotrows = int(np.sqrt(nstim))
        plotcols = int(np.ceil(nstim/plotrows))
        f, axes = plt.subplots(plotrows, plotcols, sharex=True, sharey=True)
        for ii in range(nstim):
            this_stim = stims[ii]
            if trim:
                this_stim = self.trim(this_stim)
            axes.flatten()[ii].plot(this_stim)
        f.subplots_adjust(hspace=0, wspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.setp([a.get_yticklabels() for a in f.axes[:-1]], visible=False)

    def trim(self, kernel, threshold=0.999):
        """Trims kernel to keep a fraction of the total power given by threshold.
        The center of the returned kernel is the point of the original kernel
        at which the cumulative power crosses 0.5. Expects normalized kernel."""
        squares = kernel**2
        unfolded_cumulative = np.cumsum(squares)
        midpoint = np.argmax(unfolded_cumulative > 0.5)
        if midpoint == 0 or midpoint == len(kernel):
            return kernel

        first = squares[:midpoint][::-1]
        latter = squares[midpoint:]
        l_first = len(first)
        l_latter = len(latter)
        if l_first > l_latter:
            latter = np.concatenate([latter, np.zeros([l_first-l_latter])])
        elif l_latter > l_first:
            first = np.concatenate([first, np.zeros([l_latter-l_first])])
        folded = first + latter

        cumulative = np.cumsum(folded)
        boundary = np.argmax(cumulative > threshold)
        if boundary < 1:
            boundary = 1
        start = midpoint-boundary
        end = midpoint+boundary

        return kernel[start:end]
