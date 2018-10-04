import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scisig


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


def plot_spikegram(spikes, sample_rate, markerSize=.0001):
    """adapted from https://github.com/craffel/spikegram-coding/blob/master/plotSpikeGram.py"""
    nkernels = spikes.shape[0]
    indices = np.transpose(np.nonzero(spikes))
    scalesKernelsAndOffsets = [(spikes[idx[0], idx[1]], idx[0], idx[1]) for idx in indices]

    for scale, kernel, offset in scalesKernelsAndOffsets:
        # Put a dot at each spike location.  Kernels on y axis.  Dot size corresponds to scale
        plt.plot(offset/sample_rate, nkernels-kernel, 'k.',
                 markersize=markerSize*np.abs(scale))
    plt.title("Spikegram")
    plt.xlabel("Time (s)")
    plt.ylabel("Kernel")
    plt.axis([0.0, spikes.shape[1]/sample_rate, 0.0, nkernels])
    plt.show()


def center_of_mass(signal):
    return sum(np.arange(len(signal))*signal**2)


def skewness(signal):
    com = center_of_mass(skewness)
    return sum(((np.arange(len(signal)) - com)/len(signal))**3 * signal**2)


def cf_bandwidth_plot(self, phi):
    """Each dictionary element determines a point on a plot of that element's
    bandwidth vs its center frequency."""
    centers, bandwidths = get_cf_and_bandwidth(phi)
    plt.plot(centers, bandwidths, 'b.')
    plt.xlabel('Center frequency (Hz)')
    plt.xscale('log')
    plt.ylabel('Bandwidth (Hz)')
    plt.yscale('log')
    return centers, bandwidths


def get_cf_and_bandwidth(phi, sample_rate=16000):
    spectra = np.square(np.abs(np.fft.rfft(phi, axis=1)))
    lfilter = phi.shape[1]
    freqs = np.fft.fftfreq(lfilter, d=1/sample_rate)[:spectra.shape[1]]
    centers = spectra @ freqs / spectra.sum(1)
    bandwidths = np.sqrt(spectra @ freqs**2 / spectra.sum(1) - centers**2)
    return centers, bandwidths
