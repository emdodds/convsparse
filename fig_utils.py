import numpy as np

markerSize = 8
highlight_color = [0.1, 0.7, 0.1]
kernel_samples_shown = 300
ghost_offset = 220


def signal_plot_style(ax, xticks=False):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_position('center')
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([])
    ax.set_ylim([-1, 1])
    ax.set_ylabel(ax.get_ylabel(), rotation=0,
                  ha='right', size=8, labelpad=-10)
    if not xticks:
        ax.set_xticks([])


def acts_to_numpy(acts, net):
    np_acts = np.squeeze(acts.detach().cpu().numpy())
    return np.concatenate([np.zeros([net.n_kernel, net.kernel_size-1]),
                           np_acts],
                          axis=1)


def signal_plot(sigtimes, example, ax_sig):
    ax_sig.plot(sigtimes, example)
    ax_sig.set_ylabel("Signal")


def recon(net, acts):
    return np.squeeze(net.reconstruction(acts).detach().numpy())[:-net.kernel_size+1]


def recon_plot(acts, net, sigtimes, ax_recon):
    recon0 = recon(net, acts)
    ax_recon.plot(sigtimes, recon0)
    ax_recon.set_ylabel("Reconstruction")
    return recon0


def recon_with_contrib(actspp, net, sigtimes, one_act, ax_recon1):
    recon1 = np.squeeze(net.reconstruction(actspp).detach().numpy())
    try:
        recon1 = np.concatenate([recon1, np.zeros(len(sigtimes) - len(recon1))])
    except ValueError:
        recon1 = recon1[:len(sigtimes)]
    ax_recon1.plot(sigtimes, recon1, alpha=0.5)
    recon_contrib = np.squeeze(net.reconstruction(one_act).detach().cpu().numpy())
    recon_contrib = recon_contrib[:len(sigtimes)]
    contrib_where = np.nonzero(recon_contrib)
    start = np.min(contrib_where)
    end = start + kernel_samples_shown
    ax_recon1.plot(sigtimes[start:end], recon_contrib[start:end],
                   color=highlight_color)
    ax_recon1.set_yticks([])
    ax_recon1.set_ylim([-1, 1])
    ax_recon1.set_ylabel("Reconstruction")
    return recon1


def get_new_spike(acts, actspp):
    new_ind = np.where((actspp - acts).detach().cpu().numpy())
    which = new_ind[1]
    when = new_ind[-1]
    return [[which, when]]


def kernel_plot(selected_spikes, net, sigtimes, ax_kernel, reverse=False):
    which = selected_spikes[0][0]
    ker = np.squeeze(np.squeeze(net.weights.detach().cpu().numpy())[which])
    ker = ker[:kernel_samples_shown]
    if reverse:
        ker = ker[::-1]
    ax_kernel.plot(sigtimes[:len(ker)], 2*ker, color=highlight_color, linewidth=1)
    ax_kernel.plot(sigtimes[len(ker):], np.zeros_like(sigtimes[len(ker):]), alpha=0, linewidth=1)
    ax_kernel.plot(sigtimes[ghost_offset:len(ker)+ghost_offset], 2*ker, color=highlight_color, alpha=0.25, linewidth=1)
    ax_kernel.set_ylabel("Kernel")
    ax_kernel.arrow((ghost_offset+kernel_samples_shown)/16, 0.2, 7, 0,
                    head_width=.1, head_length=1, fc='k', ec='k')


def spikegram(spikedata, sel_spikedata, net, sigtimes, ax_spikegram, extra_offset=0):
    freqs = np.logspace(np.log10(100), np.log10(6000), net.n_kernel)
    for scale, kernel, offset in spikedata:
        shifted_offset = offset + extra_offset - (net.kernel_size - 1)
        if shifted_offset < 0:
            continue
        # Put a dot at each spike location.  Kernels on y axis.  Dot size corresponds to scale
        ax_spikegram.plot(sigtimes[shifted_offset], freqs[kernel], 'k.',
                          markersize=markerSize*np.abs(scale))
    ii = 0

    for scale, kernel, offset in sel_spikedata:
        shifted_offset = offset - (net.kernel_size - 1) + extra_offset
        # Put a dot at each spike location.  Kernels on y axis.  Dot size corresponds to scale
        ax_spikegram.plot(sigtimes[shifted_offset], freqs[kernel], '.',
                          markersize=markerSize*np.abs(scale), color=highlight_color)
        ii += 1
    ax_spikegram.set_ylim([freqs[0], freqs[-1]])
    ax_spikegram.set_ylabel("Center frequency (Hz)")
    ax_spikegram.set_yscale('log')
