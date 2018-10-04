import signalset
import convsparsenet as csn
import numpy as np
import torch
import argparse
import pathlib


parser = argparse.ArgumentParser()
parser.add_argument('--seg_length', default=20000, type=int)
parser.add_argument('--lam', default=100, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--bs', default=4, type=int)
parser.add_argument('--ks', default=800, type=int)
args = parser.parse_args()

DATA_FOLDER = "/home/edodds/Data/TIMIT/"
EXP_DIR = "/home/edodds/convsparse/Results/"
EXP_SUBDIR = EXP_DIR + "csntimit_lam{}_lr{}_ks{}-000".format(args.lam, args.lr, args.ks)

data = signalset.SignalSet(data=DATA_FOLDER, all_in_memory=False)
data.seg_length = args.seg_length

net = csn.ConvSparseNet(inference_rate=50, n_iter=100, lam=args.lam,
                        initialization="minirandom", seed_length=100,
                        kernel_size=args.ks)
net.batch_size = args.bs

pathlib.Path(EXP_SUBDIR).mkdir(parents=True, exist_ok=True)

losses = []
for tt in range(1000):
    losses = losses + net.train(data, n_steps=10, learning_rate=args.lr)
    np.save(EXP_SUBDIR+"/weights.npy", np.squeeze(net.weights.detach().cpu().numpy()))
    np.save(EXP_SUBDIR+"/loss.npy", losses)
