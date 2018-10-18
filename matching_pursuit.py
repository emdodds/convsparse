import numpy as np
import torch
import torch.nn as nn
import convsparsenet

dtype = torch.float32


class MPNet(convsparsenet.ConvSparseNet):

    def __init__(self, **kwargs):
        convsparsenet.ConvSparseNet.__init__(self, **kwargs)
        self.thresh = torch.ones(self.n_kernel, device=self.device)*self.lam

    def infer(self, signal):
        with torch.no_grad():
            everything = self._infer_no_grad(signal)
        return everything

    def _infer_no_grad(self, signal):
        if not isinstance(signal, torch.Tensor):
            n_signal = len(signal)
            signal = torch.tensor(signal, device=self.device, dtype=dtype)
            signal = signal.reshape([n_signal, 1, -1])
        l_signal = signal.shape[-1]
        batch_size = signal.shape[0]

        signal = torch.cat([signal, torch.zeros([batch_size, 1,
                                                self.kernel_size-1],
                                                device=self.device)],
                           dim=2)
        resid = torch.tensor(signal)
        acts = torch.zeros(batch_size,
                           self.n_kernel,
                           l_signal,
                           device=self.device)

        cond = True
        weights = self.weights.reshape([self.n_kernel, 1, -1])
        errors = np.zeros([self.n_iter])
        L1_means = np.zeros([self.n_iter])
        for ii in range(self.n_iter):
            # note this is a cross-correlation, not convolution
            convs = nn.functional.conv1d(resid, weights)
            convmags = torch.abs(convs)
            reduced, spike_times = torch.max(convmags, dim=2)
            candidates = torch.argmax(reduced, dim=1)
            indexer = torch.arange(batch_size)
            spikes = convs[indexer, candidates,
                           spike_times[indexer, candidates]]
            spikes = (torch.abs(spikes) > self.thresh[candidates]).float()*spikes

            acts[indexer, candidates,
                 spike_times[indexer, candidates]] += spikes
            recon = self.reconstruction(acts)
            resid = signal - recon

            mse = torch.mean((signal-recon)**2)
            l1loss = torch.mean(torch.abs(acts))
            errors[ii] = mse.detach().cpu().numpy()
            L1_means[ii] = l1loss.detach().cpu().numpy()

            # print("{:4d}     mse: {:f}     l1: {:f}".format(ii, mse, l1loss))

            if torch.all(spikes == 0):
                break

        return acts, {'mse': errors, 'l1': L1_means,
                      'reconstruction': recon}

    def extra_updates(self, acts, meta):
        """Lower thresholds for dead units."""
        L1_means = torch.mean(torch.abs(acts), dim=-1)
        L1_means = torch.mean(L1_means, dim=0)
        highest = torch.max(L1_means)
        too_low = L1_means < highest/10
        self.thresh[too_low] *= 0.95
        plenty = L1_means > 0.5*highest
        self.thresh[plenty] = self.lam
