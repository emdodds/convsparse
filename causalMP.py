import numpy as np
import torch
import torch.nn as nn
import convsparsenet as csn

dtype = csn.dtype


class CausalMP(csn.ConvSparseNet):

    def __init__(self, **kwargs):
        csn.ConvSparseNet.__init__(self, **kwargs)
        self.thresh = self.lam

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
        for tt in range(l_signal):
            segment = resid[:, tt:tt+self.kernel_size]
            dots = torch.mm(segment, torch.t(weights))
            candidates = torch.argmax(torch.abs(dots), dim=1)
            spikes = dots[torch.arange(batch_size), candidates]
            # segnorm = torch.norm(segment[self.masks[candidates]])
            spikes = (torch.abs(spikes) > self.thresh).float()*spikes
            acts[torch.arange(batch_size), candidates, tt] += spikes
            resid[:, tt:tt+self.kernel_size] -= \
                spikes[:, None]*weights[candidates, :]

        padded_signal = torch.cat([signal, torch.zeros([batch_size,
                                                       self.kernel_size-1],
                                                       device=self.device)],
                                  dim=1)
        return acts, {"residual": resid,
                      "reconstruction": padded_signal - resid}

    def loss(self, signal, recon, acts):
        padded_signal = \
            torch.cat([signal, torch.zeros([signal.shape[0], 1,
                                            self.kernel_size-1],
                                           device=self.device)],
                      dim=2)
        return torch.mean((padded_signal-recon)**2)
