import time
import numpy as np
import torch
import torch.nn as nn
import convsparsenet
import utils

dtype = torch.float32


class MPNet(convsparsenet.ConvSparseNet):

    def __init__(self, adjust_thresholds=True, **kwargs):
        convsparsenet.ConvSparseNet.__init__(self, **kwargs)
        self.thresh = torch.ones(self.n_kernel, device=self.device)*self.lam
        self.adjust_thresholds = adjust_thresholds

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
        if self.adjust_thresholds:
            L1_means = torch.mean(torch.abs(acts), dim=-1)
            L1_means = torch.mean(L1_means, dim=0)
            highest = torch.max(L1_means)
            too_low = L1_means < highest/10
            self.thresh[too_low] *= 0.95
            plenty = L1_means > 0.5*highest
            self.thresh[plenty] = self.lam

    def loss(self, signal, recon, acts):
        padded_signal = \
            torch.cat([signal, torch.zeros([signal.shape[0], 1,
                                            self.kernel_size-1],
                                           device=self.device)],
                      dim=2)
        return torch.mean((padded_signal-recon)**2)


class Growing_MPNet(MPNet):

    def __init__(self, trim_threshold=0.05, **kwargs):
        self.trim_threshold = trim_threshold
        MPNet.__init__(self, **kwargs)

    def train(self, data, n_steps=1000,
              learning_rate=0.01, post_step_loss=False,
              optimizer='SGD', step_count=0,
              divide_out_signal_power=False):

        losses = []
        current_time = time.time()
        for step in range(n_steps):
            trainer = self.get_optimizer(learning_rate=learning_rate,
                                         optimizer=optimizer)
            trainer.zero_grad()
            batch = data.get_batch(batch_size=self.batch_size)
            batch = torch.tensor(batch, device=self.device, dtype=dtype)
            batch = batch.reshape([self.batch_size, 1, -1])

            acts, meta = self.infer(batch)
            recon = self.reconstruction(acts)
            training_loss = self.loss(batch, recon, acts)
            if divide_out_signal_power:
                training_loss /= torch.var(batch)
            training_loss.backward()
            trainer.step()

            self.extra_updates(acts, meta)

            loss_number = training_loss.item()
            new_time = time.time()
            elapsed = new_time - current_time
            current_time = new_time
            print(f"step: {(step + step_count):5d}   "
                  f"loss: {loss_number:f}    elapsed: {elapsed:f} sec")
            losses.append(loss_number)

        return losses

    def get_weights_tensor(self):
        self.kernel_size = int(np.max([ww.shape[-1] for ww in self.weights_list]))
        tensor = torch.zeros([1, self.n_kernel, self.kernel_size],
                             dtype=dtype, device=self.device)
        padded_weights = []
        for ii in range(self.n_kernel):
            padw = torch.cat([self.weights_list[ii],
                              torch.zeros([self.kernel_size -
                                           self.weights_list[ii].shape[-1]],
                                          device=self.device, dtype=dtype)])
            padded_weights.append(padw)
        tensor = torch.stack(padded_weights)
        return tensor.reshape([1, self.n_kernel, -1])

    def extra_updates(self, acts, meta):
        """Trim weight vectors and update dependent tensor.
        Normalize before and after trimming."""
        self.normalize_weights()
        weights = [self.trim_and_pad_kernel(ww)
                   for ww in self.weights_list]
        self.weights_list = [torch.tensor(ww, dtype=dtype,
                                          device=self.device,
                                          requires_grad=True) for ww in weights]
        self.normalize_weights()
        self.weights_list = [torch.tensor(ww, dtype=dtype,
                                          device=self.device,
                                          requires_grad=True) for ww in weights]
        self.weights = self.get_weights_tensor()

    def trim_and_pad_kernel(self, kernel_before, pad_factor=1.2):
        """Trim to threshold.
        Zero pad on each side by 10% of kernel length."""
        start, end = utils.trim_bounds(kernel_before.detach().cpu().numpy(),
                                       threshold=self.trim_threshold)
        old_padded_size = kernel_before.shape[-1]
        new_size = end-start
        new_padded_size = int(pad_factor * new_size)
        if new_padded_size > old_padded_size:
            kernel = torch.zeros([new_padded_size],
                                 device=self.device, dtype=dtype)
            left = int((new_padded_size - old_padded_size)/2)
            kernel[left:left+old_padded_size] = kernel_before
        elif new_padded_size < old_padded_size:
            left = int((old_padded_size - new_padded_size)/2)
            left = max(start, left)
            kernel = kernel_before[left:left+new_padded_size]
        else:
            kernel = kernel_before

        return kernel

    def get_initial_weights(self, initialization, seed_length):
        if seed_length != self.kernel_size:
            raise ValueError("Seed length should match kernel length for growing implementation.")
        weights = self.initial_filters(initialization, seed_length)
        weights = weights.reshape([self.n_kernel, -1])
        self.weights_list = [torch.tensor(ww, dtype=dtype,
                                          device=self.device,
                                          requires_grad=True) for ww in weights]
        return self.get_weights_tensor()

    def normalize_weights(self):
        with torch.no_grad():
            self.weights_list = [ww/torch.norm(ww, p=2, dim=-1, keepdim=True)
                                 for ww in self.weights_list]

    def get_optimizer(self, learning_rate=0.01, optimizer="SGD"):
        if optimizer == "SGD":
            return torch.optim.SGD(self.weights_list, lr=learning_rate)
        elif optimizer == "momentum":
            return torch.optim.SGD(self.weights_list, lr=learning_rate,
                                   momentum=0.9, nesterov=True)
        elif optimizer == "Adam":
            return torch.optim.Adam(self.weights_list, lr=learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported")
