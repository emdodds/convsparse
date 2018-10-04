import numpy as np
import torch
import torch.nn as nn
import time
import convsparsenet

dtype = torch.float32


class MPNet(convsparsenet.ConvSparseNet):

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
        raise NotImplementedError

        # cond = True
        # iter_count = 0
        # while cond:
        #     convs = 
        #     convmags = np.abs(convs)
        #     accept = False
        #     while not accept:
        #         winner = np.unravel_index(convmags.argmax(), convs.shape)
        #         coeffswinner = [winner[0], winner[1], winner[3]]
        #         coeffswinner[1] += self.lfilter - 1
        #         coeffswinner = tuple(coeffswinner)
        #         if coeffs[coeffswinner] == 0 or convmags[winner] == 0:
        #             accept = True
        #         else:
        #             convmags[winner] = 0

        #     spike = convs[winner]
        #     iter_count += 1
        #     if np.abs(spike) < self.min_spike or iter_count > self.max_iter:
        #         cond = False
        #     if cond:
        #         coeffs[coeffswinner] = convs[winner]
        #         feed_dict = {d['x']: signal,
        #                      d['coeffs']: coeffs}
        #         resid, mse, xhat = sess.run([d['resid'], d['mse'], d['xhat']],
        #                                     feed_dict=feed_dict)
        #         errors.append(mse)

            return acts, {'loss': losses, 'l1': l1_means}