import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scisig
from scipy.optimize import curve_fit
from scipy.io import wavfile

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


def full_gammachirp(t, a, b, c, f, phi):
    """
    n = 4 gammachirp as in Irino and Patterson 1997
    t is in seconds, f is in 1/seconds, other quantities dimensionless
    """
    ERB = 24.7 + 0.108*f
    return a*t**3 * np.exp(-2*np.pi*b*ERB*t) * \
        np.cos(2*np.pi*f*t + c*np.log(1e-6 + t) + phi)


def zero_padded_gammachirp(t, a, b, c, f, phi, t0):
    """Pads gammachirp with zeros on steep side."""
    return (t > t0)*np.nan_to_num(full_gammachirp(t, a, b, c, f, phi))


def fit_gammachirp(times, kernel):
    return curve_fit(zero_padded_gammachirp, times, kernel,
                     p0=[1, 1, 1, 1, 0, 0])


def fit_gammachirp_no_offset(times, kernel):
    # TODO: start with estimate of freq from fft, a from normalization; need good starting point
    return curve_fit(full_gammachirp, times, kernel, p0=[1, 1, 1, 1, 0])


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


def plot_spikegram(spikes, sample_rate, markerSize=.0001, ax=None):
    """adapted from https://github.com/craffel/spikegram-coding/blob/master/plotSpikeGram.py"""
    nkernels = spikes.shape[0]
    indices = np.transpose(np.nonzero(spikes))
    scalesKernelsAndOffsets = [(spikes[idx[0], idx[1]], idx[0], idx[1]) for idx in indices]

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for scale, kernel, offset in scalesKernelsAndOffsets:
        # Put a dot at each spike location.  Kernels on y axis.  Dot size corresponds to scale
        ax.plot(offset/sample_rate, nkernels-kernel, 'k.',
                 markersize=markerSize*np.abs(scale))
    ax.set_title("Spikegram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Kernel")
    ax.set_xlim([0.0, spikes.shape[1]/sample_rate])
    ax.set_ylim([0.0, nkernels])


def center_of_mass(signal):
    return sum(np.arange(len(signal))*signal**2)


def skewness(signal):
    com = center_of_mass(skewness)
    return sum(((np.arange(len(signal)) - com)/len(signal))**3 * signal**2)


def cf_bandwidth_plot(phi, bw_type="std", ax=None):
    """Each dictionary element determines a point on a plot of that element's
    bandwidth vs its center frequency."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    centers, bandwidths = get_cf_and_bandwidth(phi, bw_type=bw_type)
    ax.plot(centers, bandwidths, 'b.')
    ax.set_xlabel('Center frequency (Hz)')
    ax.set_xscale('log')
    ax.set_ylabel('Bandwidth (Hz)')
    ax.set_yscale('log')
    return centers, bandwidths


def get_cf_and_bandwidth(phi, sample_rate=16000, bw_type="std"):
    spectra = np.square(np.abs(np.fft.rfft(phi, axis=1)))
    lfilter = phi.shape[1]
    freqs = np.fft.fftfreq(lfilter, d=1/sample_rate)[:spectra.shape[1]]
    centers = spectra @ freqs / spectra.sum(1)
    if bw_type == "std":
        bandwidths = np.sqrt(spectra @ freqs**2 / spectra.sum(1) - centers**2)
    elif bw_type == "length":
        bandwidths = np.array([sample_rate/len(trim(kernel, 0.99)) for kernel in phi])
    elif bw_type == "3db":
        peaks = np.max(spectra, axis=1)
        masks = spectra > 0.5*peaks[:, None]
        lefts = np.argmax(masks, axis=1)
        rights = np.argmax(masks[:, ::-1], axis=1)
        bandwidths = freqs[-rights] - freqs[lefts]
    else:
        raise ValueError("Not supported: {}".format(bw_type))
    return centers, bandwidths


def write_sound(filename, signal, sample_rate):
    signal /= np.max(signal)
    wavfile.write(filename, sample_rate, signal)


def tiled_plot(stims, trim=False):
    """Tiled plots of the given signals. Zeroth index is which signal.
    Kind of slow, expect about 10s for 100 plots."""
    nstim = stims.shape[0]
    plotrows = int(np.sqrt(nstim))
    plotcols = int(np.ceil(nstim/plotrows))
    f, axes = plt.subplots(plotrows, plotcols,
                           sharex=(not trim), sharey=True)
    for ii in range(nstim):
        this_stim = stims[ii]
        if trim:
            this_stim = trim(this_stim, threshold=trim)
        axes.flatten()[ii].plot(this_stim)
    f.subplots_adjust(hspace=0, wspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.setp([a.get_yticklabels() for a in f.axes[:-1]], visible=False)


def trim(kernel, threshold=0.999):
    """Trims kernel to keep a fraction of the total power given by threshold.
    The center of the returned kernel is the point of the original kernel
    at which the cumulative power crosses 0.5. Expects normalized kernel."""
    if not isinstance(threshold, float):
        threshold = 0.999
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
    start = max(0, midpoint-boundary)
    end = min(len(kernel), midpoint+boundary)

    return kernel[start:end]


def show_spectra(phi):
    """Show a tiled plot of the power spectra of the given dictionary."""
    spectra = np.square(np.abs(np.fft.rfft(phi, axis=1)))
    tiled_plot(spectra)


def sort_by_freq(phi):
    centerfreqs, _ = get_cf_and_bandwidth(phi)
    sorter = np.argsort(centerfreqs)
    return phi[sorter]