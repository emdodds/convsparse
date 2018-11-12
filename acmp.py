import numpy as np
import torch
import torch.nn as nn
import convsparsenet as csn

dtype = csn.dtype


class ACMP(csn.ConvSparseNet):

    def __init__(self, adapt_factor=2., adjust_thresholds=True, **kwargs):
        csn.ConvSparseNet.__init__(self, **kwargs)
        self.thresh = torch.ones(self.n_kernel, device=self.device)*self.lam
        self.set_decay_times()
        self.adapt_factor = adapt_factor
        self.adjust_thresholds = adjust_thresholds

    def infer(self, signal):
        with torch.no_grad():
            everything = self._infer_no_grad(signal)
        return everything

    def _infer_no_grad(self, signal):
        n_signal = signal.shape[0] if len(signal.shape) > 1 else 1
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, device=self.device, dtype=dtype)
        signal = signal.reshape([n_signal, -1])
        l_signal = signal.shape[-1]
        batch_size = signal.shape[0]
        acts = torch.zeros(batch_size,
                           self.n_kernel,
                           l_signal,
                           device=self.device,
                           requires_grad=False)
        resid = torch.cat([signal, torch.zeros([batch_size,
                                               self.kernel_size-1],
                                               device=self.device)],
                          dim=1)

        weights = self.weights.detach().reshape(self.n_kernel, -1)
        thresh = self.thresh.repeat([batch_size, 1])
        for tt in range(l_signal):
            segment = resid[:, tt:tt+self.kernel_size]
            dots = torch.mm(segment, torch.t(weights))
            candidates = torch.argmax(torch.abs(dots), dim=1)
            spikes = dots[torch.arange(batch_size), candidates]
            abspikes = torch.abs(spikes)

            indexer = torch.arange(batch_size)
            raw_condition = (abspikes > thresh[indexer, candidates]).float()

            spikes = raw_condition*spikes
            acts[indexer, candidates, tt] += spikes
            resid[:, tt:tt+self.kernel_size] -= \
                spikes[:, None]*weights[candidates, :]

            # adapt threshold of spiking unit
            thresh[indexer, candidates] += \
                self.adapt_factor * raw_condition * self.thresh[candidates]
            # decay all thresholds towards init
            thresh += -(thresh - self.thresh)/self.decay_time
            # thresh = (thresh < self.thresh).float()*self.thresh[None, :] + \
            #          (thresh >= self.thresh).float()*thresh

        padded_signal = torch.cat([signal, torch.zeros([batch_size,
                                                       self.kernel_size-1],
                                                       device=self.device)],
                                  dim=1)
        return acts, {"residual": resid,
                      "reconstruction": padded_signal - resid}

    def extra_updates(self, acts, meta):
        """Lower thresholds for dead units."""
        if self.adjust_thresholds:
            L1_means = torch.mean(torch.abs(acts), dim=-1)
            L1_means = torch.mean(L1_means, dim=0)
            highest = torch.max(L1_means)
            too_low = L1_means < highest/10
            self.thresh[too_low] *= 0.95
            plenty = L1_means > 0.5*highest
            self.thresh[plenty] = self.lam
        self.set_decay_times()

    def set_decay_times(self):
        weights = self.weights.detach().cpu().numpy().reshape([self.n_kernel, -1])
        spectra = np.square(np.abs(np.fft.rfft(weights, axis=1)))
        lfilter = weights.shape[1]
        freqs = np.fft.fftfreq(lfilter, d=1)[:spectra.shape[1]]
        centers = spectra @ freqs / spectra.sum(1)
        self.decay_time = torch.tensor(1./centers, device=self.device,
                                       dtype=torch.float32)

    def loss(self, signal, recon, acts):
        padded_signal = \
            torch.cat([signal, torch.zeros([signal.shape[0], 1,
                                            self.kernel_size-1],
                                           device=self.device)],
                      dim=2)
        return torch.mean((padded_signal-recon)**2)