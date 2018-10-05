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
        self.weights = torch.tensor(weights, dtype=dtype, device=self.device, requires_grad=True)

    def initial_filters(self, initialization="minirandom", seed_length=100):
        """If 1D, Return either a set of gammachirp filters or random filters,
        with seed_length iid normal samples in the middle surrounded by 0s,
        normalized. Otherwise return Gaussian noise, not normalized."""
        if initialization == "gammachirp":
            gammachirps = np.zeros([self.n_kernel, self.kernel_size])
            freqs = np.logspace(np.log10(100), np.log10(6000), self.n_kernel)
            times = np.linspace(0, self.kernel_size/16000,
                                self.kernel_size)
            for ii in range(self.n_kernel):
                gammachirps[ii] = dcGC(times, freqs[ii])
            filters = gammachirps
        elif initialization == "minifourier":
            timepoints = np.arange(self.kernel_size)
            frequencies = [(nn+1)/(2*seed_length) for nn in range(self.n_kernel)]
            filters = [np.sin(2*np.pi*freq*timepoints) for freq in frequencies]
            filters = np.array(filters)
            start = int(self.kernel_size/2 - seed_length/2)
            end = start + seed_length
            filters[:, :start] = 0
            filters[:, end:] = 0
            filters -= filters.mean(axis=1, keepdims=True)
        elif initialization == "minirandom":
            filters = np.random.randn(self.n_kernel, self.kernel_size)
            start = int(self.kernel_size/2 - seed_length/2)
            end = start + seed_length
            filters[:, :start] = 0
            filters[:, end:] = 0
        else:
            raise ValueError("Unsupported initialization")
        filters /= np.sqrt(np.sum(filters**2, axis=1))[:, None]
        return filters.reshape(filters.shape+(1,))

    def loss(self, signal, recon, acts):
        if recon.shape[-1] > signal.shape[-1]:
            padding = torch.zeros(list(signal.shape[:-1]) +
                                  [recon.shape[-1] - signal.shape[-1]])
            signal = torch.cat([padding, signal], dim=-1)
        mse = torch.mean((signal-recon)**2)
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

    def train(self, data, n_steps=1000, learning_rate=0.01, post_step_loss=False):
        trainer = torch.optim.SGD([self.weights], lr=learning_rate)
                                  #momentum=0.9, nesterov=True)

        losses = []
        current_time = time.time()
        for step in range(n_steps):
            trainer.zero_grad()
            #self.weights.requires_grad = True
            batch = data.get_batch(batch_size=self.batch_size)
            batch = torch.tensor(batch, device=self.device, dtype=dtype)
            batch = batch.reshape([self.batch_size, 1, -1])

            acts, meta = self.infer(batch)
            # acts = acts.detach()
            recon = self.reconstruction(acts)
            training_loss = self.loss(batch, recon, acts)
            training_loss.backward()
            trainer.step()

            with torch.no_grad():
                self.weights /= torch.norm(self.weights, p=2,
                                           dim=-1, keepdim=True)

            loss_number = training_loss.item()
            new_time = time.time()
            elapsed = new_time - current_time
            current_time = new_time
            print(f"step: {step:5d}   loss: {loss_number:f}    elapsed: {elapsed:f} sec")
            losses.append(loss_number)

            if post_step_loss:
                acts, _ = self.infer(batch)
                training_loss = self.loss(batch, self.reconstruction(acts), acts)
                print(f"loss on same batch after step: {training_loss.item():f}")

        return losses

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
        recon = np.squeeze(self.reconstruction(acts).detach().numpy())
        ax.plot(times, recon)
        ax.set_title('Reconstruction')

        np_acts = np.squeeze(acts.detach().numpy())
        np_acts = np.concatenate([np.zeros([self.n_kernel, self.kernel_size-1]),
                                  np_acts],
                                 axis=1)
        utils.plot_spikegram(np_acts,
                             sample_rate=sample_rate, markerSize=1, ax=axes[2])
        print("Signal-noise ratio: {:f} dB".format(utils.snr(padded_signal, recon)))
        return acts, meta

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

    # def fast_sort(self, measure="L0", plot=False, savestr=None):
    #     """Sorts filters by moving average usage of specified type, or by center frequency.
    #     Options for measure: L0, L1, f. L0 by default."""
    #     if measure == "f" or measure == "frequency":
    #         usages, _ = self.get_cf_and_bandwidth()
    #     elif measure == "L1":
    #         usages = self.L1acts
    #     else:
    #         usages = self.L0acts
    #     sorter = np.argsort(usages)
    #     self.sort(usages, sorter, plot, savestr)
    #     return usages[sorter]

    # def sort(self, usages, sorter, plot, savestr=None):
    #     self.weights = self.weights[sorter]
    #     self.L0acts = self.L0acts[sorter]
    #     self.L1acts = self.L1acts[sorter]
    #     self.L2acts = self.L2acts[sorter]
    #     if plot:
    #         plt.figure()
    #         plt.plot(usages[sorter])
    #         plt.title('L0 Usage')
    #         plt.xlabel('Dictionary index')
    #         plt.ylabel('Fraction of stimuli')
    #         if savestr is not None:
    #             plt.savefig(savestr, format='png', bbox_inches='tight')