import numpy as np
import torch
import torch.nn as nn
import convsparsenet as csn

dtype = csn.dtype


def roll(x, shift, dim=-1, fill_pad=None):
    if 0 == shift:
        return x

    elif shift < 0:
        shift = -shift
        gap = x.index_select(dim, torch.arange(shift))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([x.index_select(dim, torch.arange(shift, x.size(dim))), gap], dim=dim)

    else:
        shift = x.size(dim) - shift
        gap = x.index_select(dim, torch.arange(shift, x.size(dim)))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([gap, x.index_select(dim, torch.arange(shift))], dim=dim)


class FilterThreshold(csn.ConvSparseNet):

    def __init__(self, thresh, **kwargs):
        csn.ConvSparseNet.__init__(self, **kwargs)
        self.thresh = thresh

    def infer(self, signal):
        # with torch.no_grad():
        #     everything = self._infer_no_grad(signal)
        # return everything
        return self._infer_no_grad(signal)

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

        weights = self.weights.reshape([self.n_kernel, 1, -1])
        xcorr = nn.functional.conv1d(signal, weights)
        rollforward = roll(xcorr, 1)
        rollbackward = roll(xcorr, -1)
        peaks = xcorr * (xcorr > rollforward).float() \
                      * (xcorr > rollbackward).float()

        acts = peaks * (peaks > self.thresh).float()

        return acts, {}

    def loss(self, signal, recon, acts):
        padded_signal = \
            torch.cat([signal, torch.zeros([signal.shape[0], 1,
                                            self.kernel_size-1],
                                           device=self.device)],
                      dim=2)

        mse = torch.mean((padded_signal-recon)**2)
        return mse + self.lam*torch.mean(torch.abs(acts))