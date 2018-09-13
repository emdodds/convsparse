import numpy as np
import torch
import torch.nn as nn
import time


dtype = torch.float32


class ConvSparseNet():
    def __init__(self, n_kernel=32, kernel_size=800, lam=0.4, n_iter=200, batch_size=32,
                 inference_rate=0.5, device=torch.device("cuda:0")):
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
        self.device

        weights = np.random.normal(size=[1, n_kernel, kernel_size])
        weights /= np.linalg.norm(weights, axis=-1, keepdims=True)
        self.weights = torch.tensor(weights, dtype=dtype, device=self.device)

    def loss(self, signal, recon, acts):
        mse = nn.MSELoss()(recon, signal)
        l1loss = torch.mean(torch.abs(acts))
        return torch.add(mse, self.lam*l1loss)

    def reconstruction(self, acts):
        return nn.functional.conv1d(acts, self.weights)

    def infer(self, signal):
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
            l1_means.append(torch.mean(torch.abs(acts)))
            optimizer.step()

        return acts, {'loss': losses, 'l1': l1_means}

    def train(self, data, n_steps=1000):
        trainer = torch.optim.SGD([self.weights], lr=0.01,
                                  momentum=0.9, nesterov=True)

        current_time = time.time()
        for step in range(n_steps):
            batch = data.get_batch(batch_size=self.batch_size)
            acts, _ = self.infer(batch)
            training_loss = self.loss(batch, self.reconstruction(acts), acts)
            training_loss.backward()
            trainer.step()
            loss_number = training_loss.item()
            new_time = time.time()
            elapsed = new_time - current_time
            current_time = new_time
            print(f"step: {step}   loss: {loss_number:f}    elapsed: {elapsed} sec")
