import numpy as np
import torch
import torch.nn as nn
import time


dtype = torch.float32


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


class ConvSparseNet():
    def __init__(self, n_kernel=32, kernel_size=800, lam=0.4, n_iter=200, batch_size=32,
                 inference_rate=0.5,
                 weight_decay=0.01,
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

        self.weight_decay = weight_decay

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
        mse = torch.mean((signal-recon)**2)
        l1loss = torch.mean(torch.abs(acts))
        return torch.add(mse, self.lam*l1loss)

    def reconstruction(self, acts):
        return nn.functional.conv1d(acts, self.weights)

    def infer(self, signal):
        if not isinstance(signal, torch.Tensor):
            n_signal = len(signal)
            signal = torch.tensor(signal, device=self.device, dtype=dtype)
            signal = signal.reshape([n_signal, 1, -1])
        l_signal = signal.shape[-1]
        batch_size = signal.shape[0]
        acts = torch.zeros(batch_size,
                           self.n_kernel,
                           l_signal+(self.kernel_size - 1),
                           device=self.device)
        acts.requires_grad = True

        optimizer = torch.optim.SGD([acts], lr=self.inference_rate)
        losses = []
        l1_means = []
        for ii in range(self.n_iter):
            optimizer.zero_grad()
            total_loss = self.loss(signal, self.reconstruction(acts), acts)
            total_loss.backward()
            losses.append(total_loss.item())
            l1_means.append(torch.mean(torch.abs(acts)).item())
            optimizer.step()

        return acts, {'loss': losses, 'l1': l1_means}

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

            acts, _ = self.infer(batch)

            training_loss = self.loss(batch, self.reconstruction(acts), acts)
            # loss_with_reg = training_loss + self.weight_decay*torch.norm(self.weights, p=2)
            # loss_with_reg.backward()
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
