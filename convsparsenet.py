import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import utils


dtype = torch.float32


class ConvSparseNet():
    def __init__(self,
                 n_kernel=32,
                 kernel_size=800,
                 lam=100,
                 n_iter=200,
                 batch_size=32,
                 inference_rate=50,
                 initialization="minirandom",
                 seed_length=100,
                 device=torch.device("cuda:0")):
        """
        Args:
        n_kernel        : (int) how many convolutional kernels
        kernel_size     : (int) length in time steps of each kernel
        lam             : (float) sparsity parameter lambda
        n_iter          : (float) number of inference time steps
        batch_size      : (int) default size of batches for learning
        inference_rate  : (float) length of each inference time step
        device          : e.g., torch.device('cuda:0') or torch.device('cpu')
        """
        self.n_kernel = n_kernel
        self.kernel_size = kernel_size
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.inference_rate = inference_rate
        self.lam = lam
        self.device = device

        weights = self.initial_filters(initialization, seed_length)
        weights = weights.reshape([1, n_kernel, kernel_size])
        self.weights = torch.tensor(weights, dtype=dtype,
                                    device=self.device, requires_grad=True)

    def initial_filters(self, initialization="minirandom", seed_length=100):
        """If 1D, Return either a set of gammachirp filters or random filters,
        with seed_length iid normal samples in the middle surrounded by 0s,
        normalized. Otherwise return Gaussian noise, not normalized."""
        if "gammachirp" in initialization:
            gammachirps = np.zeros([self.n_kernel, self.kernel_size])
            freqs = np.logspace(np.log10(100), np.log10(6000), self.n_kernel)
            times = np.linspace(0, self.kernel_size/16000,
                                self.kernel_size)
            for ii in range(self.n_kernel):
                gammachirps[ii] = utils.dcGC(times, freqs[ii])
            filters = gammachirps
        elif "fourier" in initialization:
            timepoints = np.arange(self.kernel_size)
            frequencies = [(nn+1)/(2*seed_length) for nn in range(self.n_kernel)]
            filters = [np.sin(2*np.pi*freq*timepoints) for freq in frequencies]
            filters = np.array(filters)
            start = int(self.kernel_size/2 - seed_length/2)
            end = start + seed_length
            filters[:, :start] = 0
            filters[:, end:] = 0
            filters -= filters.mean(axis=1, keepdims=True)
        elif "minirandom" in initialization:
            filters = np.random.randn(self.n_kernel, self.kernel_size)
            start = int(self.kernel_size/2 - seed_length/2)
            end = start + seed_length
            filters[:, :start] = 0
            filters[:, end:] = 0
        else:
            raise ValueError("Unsupported initialization")
        if "_r" in initialization:
            filters = np.flip(filters, axis=-1).copy()
        filters /= np.sqrt(np.sum(filters**2, axis=1))[:, None]
        return filters.reshape(filters.shape+(1,))

    def mse(self, signal, recon, acts):
        padded_signal = \
            torch.cat([signal, torch.zeros([signal.shape[0], 1,
                                            self.kernel_size-1],
                                           device=self.device)],
                      dim=2)
        return torch.mean((padded_signal-recon)**2)

    def loss(self, signal, recon, acts):
        mse = self.mse(signal, recon, acts)
        l1loss = torch.mean(torch.abs(acts))
        return torch.add(mse, self.lam*l1loss)

    def reconstruction(self, acts):
        """Return the stimulus reconstruction given by the convolution of
        the current weights with the given activations. Notice this is a true
        convolution, NOT a cross-correlation."""
        return nn.functional.conv1d(acts, torch.flip(self.weights, [2]),
                                    padding=self.kernel_size-1)

    def infer(self, signal):
        if not isinstance(signal, torch.Tensor):
            n_signal = len(signal)
            signal = torch.tensor(signal, device=self.device, dtype=dtype)
            signal = signal.reshape([n_signal, 1, -1])
        l_signal = signal.shape[-1]
        batch_size = signal.shape[0]
        acts = torch.zeros(batch_size,
                           self.n_kernel,
                           l_signal,
                           device=self.device)
        acts.requires_grad = True

        optimizer = torch.optim.SGD([acts], lr=self.inference_rate)
        losses = []
        l1_means = []
        for ii in range(self.n_iter):
            optimizer.zero_grad()
            recon = self.reconstruction(acts)
            total_loss = self.loss(signal, recon, acts)
            total_loss.backward()
            losses.append(total_loss.item())
            l1_means.append(torch.mean(torch.abs(acts)).item())
            optimizer.step()

        return acts, {'loss': losses, 'l1': l1_means,
                      'reconstruction': recon}

    def train(self, data, n_steps=1000,
              learning_rate=0.01, post_step_loss=False,
              optimizer='SGD', step_count=0,
              divide_out_signal_power=False):
        if optimizer == "SGD":
            trainer = torch.optim.SGD([self.weights], lr=learning_rate)
        elif optimizer == "momentum":
            trainer = torch.optim.SGD([self.weights], lr=learning_rate,
                                      momentum=0.9, nesterov=True)
        elif optimizer == "Adam":
            trainer = torch.optim.Adam([self.weights], lr=learning_rate)

        losses = []
        current_time = time.time()
        for step in range(n_steps):
            trainer.zero_grad()
            #self.weights.requires_grad = True
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

            with torch.no_grad():
                self.weights /= torch.norm(self.weights, p=2,
                                           dim=-1, keepdim=True)

            loss_number = training_loss.item()
            new_time = time.time()
            elapsed = new_time - current_time
            current_time = new_time
            print(f"step: {(step + step_count):5d}   "
                  f"loss: {loss_number:f}    elapsed: {elapsed:f} sec")
            losses.append(loss_number)

            if post_step_loss:
                acts, _ = self.infer(batch)
                training_loss = self.loss(batch, self.reconstruction(acts), acts)
                print(f"loss on same batch after step: {training_loss.item():f}")

        return losses

    def extra_updates(self, acts, meta):
        pass

    def test_inference(self, signal, sample_rate=16000):
        if len(signal.shape) < 2:
            length = signal.shape[0]
        else:
            length = signal.shape[1]
        acts, meta = self.infer(signal)

        signal = np.squeeze(signal)
        padded_signal = np.concatenate([signal, np.zeros(self.kernel_size-1)])
        padded_length = len(padded_signal)
        times = np.arange(padded_length) / sample_rate
        fig, axes = plt.subplots(3, 1, sharex=True)
        ax = axes[0]
        ax.plot(times, padded_signal)
        ax.set_title('Original signal')
        ax = axes[1]
        recon = np.squeeze(self.reconstruction(acts).detach().cpu().numpy())
        ax.plot(times, recon)
        ax.set_title('Reconstruction')

        np_acts = np.squeeze(acts.detach().cpu().numpy())
        np_acts = np.concatenate([np.zeros([self.n_kernel, self.kernel_size-1]),
                                  np_acts],
                                 axis=1)
        utils.plot_spikegram(np_acts,
                             sample_rate=sample_rate, markerSize=1, ax=axes[2])
        print("Signal-noise ratio: {:f} dB".format(utils.snr(padded_signal, recon)))
        return acts, meta

    def evaluate(self, test_set):
        with torch.no_grad():
            everything = self._evaluate(test_set)
        return everything

    def _evaluate(self, test_set):
        n_signal = len(test_set)
        n_batches = int(n_signal/self.batch_size)
        if n_batches != n_signal/self.batch_size:
            n_batches += 1
        errors = []
        l1means = []
        counts = []
        for bb in range(n_batches):
            signals = test_set[self.batch_size*bb:self.batch_size*(bb+1)]
            acts, meta = self.infer(signals)
            recon = self.reconstruction(acts)
            normed_error = self.mse(signals, recon, acts)/torch.mean(signals**2)
            errors.append(normed_error.detach().cpu().numpy())
            l1means.append(torch.mean(torch.abs(acts)).detach().cpu().numpy())
            counts.append(np.count_nonzero(acts.detach().cpu().numpy())
                          / np.prod(signals.shape))
            print("Finished batch {} of {}".format(bb+1, n_batches))
        cat_errors = np.array(errors)
        cat_l1 = np.array(l1means)
        cat_counts = np.array(counts)

        results = {}
        results["error"] = np.mean(cat_errors)
        results["error_std"] = np.std(cat_errors)
        results["l1"] = np.mean(cat_l1)
        results["l1_std"] = np.std(cat_l1)
        results["count"] = np.mean(cat_counts)
        results["count_std"] = np.std(cat_counts)
        return results

    def revcorr(self, nstims=10, delay=100):
        RFs = np.zeros((self.n_kernel, self.kernel_size+delay))
        spikecounts = np.zeros(self.nunits)
        for nn in range(nstims):
            signal = np.random.normal(size=40000)
            lsignal = signal.shape[0]
            acts, _ = self.infer(signal)
            recon = self.reconstruction(acts)
            for tt in range(self.kernel_size, lsignal-delay):
                segment = signal[tt-self.kernel_size:tt+delay]
                RFs += np.outer(acts[:, tt], segment)
                spikecounts += acts[:, tt]
        RFs = RFs/(np.linalg.norm(RFs, axis=1)[:, None])
        return RFs, spikecounts

    def load(self, path):
        weights = np.load(path + "/weights.npy")
        weights = weights.reshape([1, self.n_kernel, self.kernel_size])
        self.weights = torch.tensor(weights, dtype=dtype,
                                    device=self.device, requires_grad=True)
