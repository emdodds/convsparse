import os
from scipy.io import wavfile
from scipy import signal as scisig
import numpy as np
import matplotlib.pyplot as plt
import utils


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
                 seg_length=80000,
                 all_in_memory=True,
                 norm_factor=500,
                 split=None):
        self.sample_rate = sample_rate
        self.min_length = min_length
        self.seg_length = seg_length
        self.all_in_memory = all_in_memory
        self.norm_factor = norm_factor
        self.split = split
        if isinstance(data, str):
            self.load_from_folder(data)
        else:
            self.data = data
            self.ndata = len(data)

    def load_file(self, filename):
        rate, signal = wavfile.read(filename)
        if rate != self.sample_rate:
            raise NotImplementedError('The signal in ' + filename +
                                      ' does not match the given' +
                                      ' sample rate.')
        if signal.shape[0] > self.min_length:
            # bandpass
            signal = signal/signal.std()
            signal = butter_bandpass_filter(signal, lowcut, highcut,
                                            self.sample_rate, order=5)
        if self.norm_factor <= 0:
            signal /= np.max(signal)
        return signal

    def load_from_folder(self, folder="/home/edodds/Data/TIMIT/", cache=True):
        files = os.listdir(folder)
        if 'cache.npy' in files and self.all_in_memory:
            self.data = np.load(os.path.join(folder, 'cache.npy'))
            self.ndata = len(self.data)
            print("Loaded data from cache.npy")
            return
        self.data = []
        for ff in files:
            if self._check_filename(ff):
                file = os.path.join(folder, ff)
                if self.all_in_memory:
                    signal = self.load_file(file)
                    self.data.append(signal)
                else:
                    self.data.append(file)
        self.ndata = len(self.data)
        print("Found ", self.ndata, " files")
        if cache and self.all_in_memory:
            np.save(os.path.join(folder, 'cache.npy'), self.data)

    def _check_filename(self, filename):
        """True iff the file ends in .wav and is in the (test/train) split if specified."""
        condition = filename.endswith('.wav')
        if self.split is not None:
            condition = condition and self.split in filename
        return condition

    def rand_stim(self):
        """Get one random signal."""
        not_found = True
        while not_found:
            which = np.random.randint(low=0, high=self.ndata)
            if self.all_in_memory:
                signal = self.data[which]
            else:
                signal = self.load_file(self.data[which])
            excess = signal.shape[0] - self.seg_length
            if excess >= 0:
                where = np.random.randint(low=0, high=excess)
                segment = signal[where:where+self.seg_length]
                not_found = False
        if self.norm_factor > 0:
            segment = (self.norm_factor*len(segment)/self.sample_rate) * segment / np.linalg.norm(segment)
        return segment

    def get_batch(self, batch_size):
        return [self.rand_stim() for _ in range(batch_size)]

    def write_sound(self, filename, signal):
        signal /= np.max(signal)
        wavfile.write(filename, self.sample_rate, signal)

    def tiled_plot(self, stims, trim=False):
        """Tiled plots of the given signals. Zeroth index is which signal.
        Kind of slow, expect about 10s for 100 plots."""
        utils.tiled_plot(stims, trim=trim)

    def trim(self, kernel, threshold=0.999):
        """Trims kernel to keep a fraction of the total power given by threshold.
        The center of the returned kernel is the point of the original kernel
        at which the cumulative power crosses 0.5. Expects normalized kernel."""
        return utils.trim(kernel, threshold=threshold)

    def show_spectra(self, phi):
        """Show a tiled plot of the power spectra of the given dictionary."""
        utils.show_spectra(phi)
