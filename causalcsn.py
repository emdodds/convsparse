import numpy as np
import torch
import torch.nn as nn
import convsparsenet as csn

dtype = csn.dtype


class CausalConvSparseNet(csn.ConvSparseNet):

    def infer(self, signal):
        # scheme: create trainable acts tensor for relevant window, train on windowed residual
        # then advance the window, saving acts, and updating residual for out-of-window acts
        if not isinstance(signal, torch.Tensor):
            n_signal = len(signal)
            signal = torch.tensor(signal, device=self.device, dtype=dtype)
            signal = signal.reshape([n_signal, 1, -1])
        l_signal = signal.shape[-1]
        batch_size = signal.shape[0]
        resid = torch.cat([signal, torch.zeros([batch_size, 1,
                                               self.kernel_size-1],
                                               device=self.device,
                                               dtype=dtype)],
                          dim=-1)
        resid.requires_grad = False
        acts = torch.zeros(batch_size,
                           self.n_kernel,
                           l_signal,
                           device=self.device,
                           requires_grad=False)

        histories = {"loss": [], "l1": []}
        for tt in range(l_signal):
            segment = torch.cat([resid[:, :, tt:tt+self.kernel_size],
                                 torch.zeros([batch_size, 1,
                                              self.kernel_size-1],
                                             device=self.device,
                                             dtype=dtype)],
                                dim=-1)
            temp_acts = torch.tensor(acts[:, :, tt:tt+self.kernel_size].detach(),
                                     device=self.device,
                                     requires_grad=True)
            optimizer = torch.optim.SGD([temp_acts], lr=self.inference_rate)
            if tt > l_signal - self.kernel_size:
                # pad with frozen zeros
                print("padding")
                temp_acts = self.pad_to_kernel_size(temp_acts, batch_size)
            losses = []
            l1_means = []
            for _ in range(self.n_iter):
                optimizer.zero_grad()
                recon = self.reconstruction(temp_acts)
                loss = self.loss(segment[:, :, :self.kernel_size],
                                 recon, temp_acts)
                loss.backward()
                losses.append(loss.item())
                l1_means.append(torch.mean(torch.abs(temp_acts)).item())
                optimizer.step()
            histories["loss"].append(losses)
            histories["l1"].append(l1_means)
            acts[:, :, tt:tt+self.kernel_size] = temp_acts.detach()
            resid[:, :, tt:tt+self.kernel_size] -= \
                torch.matmul(acts[:, :, tt], self.weights).detach()
            print(tt)

        return acts, histories

    def pad_to_kernel_size(self, segment, batch_size):
        return torch.cat([segment,
                          torch.zeros([batch_size, 1,
                                       self.kernel_size-segment.shape[-1]],
                                      device=self.device, dtype=dtype)])
