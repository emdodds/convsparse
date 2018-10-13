import numpy as np
import torch
import argparse
import pathlib
import json

import signalset
import convsparsenet as csn
import causalMP
# import matching_pursuit

parser = argparse.ArgumentParser()
parser.add_argument('--c', default=None, type=str)
parser.add_argument('--device', default="cpu", type=str)
args = parser.parse_args()

if args.c is not None:
    with open(args.c, 'r') as fh:
        config = json.load(fh)
else:
    from config import config

data = signalset.SignalSet(data=config["data_folder"],
                           all_in_memory=False)
data.seg_length = args.seg_length

if args.device == "gpu":
    device = torch.device("cuda:0")
else:
    device = torch.device(args.device)

if config["model"] == "csn":
    net = csn.ConvSparseNet(inference_rate=50, n_iter=100,
                            lam=config["sparseness_parameter"],
                            initialization="minirandom", seed_length=100,
                            kernel_size=config["kernel_size"],
                            device=device)
elif config["model"] == "causal":
    net = causalMP.CausalMP(device=device,
                            thresh=0.1, seed_length=800)
else:
    raise ValueError("Unsupported model specifiction: {}".format(config["model"]))

net.batch_size = args.bs

EXP_SUBDIR = config["experiment_folder"]
pathlib.Path(EXP_SUBDIR).mkdir(parents=True, exist_ok=True)

losses = []
for tt in range(1000):
    losses = losses + net.train(data, n_steps=10, learning_rate=args.lr)
    np.save(EXP_SUBDIR+"/weights.npy", np.squeeze(net.weights.detach().cpu().numpy()))
    np.save(EXP_SUBDIR+"/loss.npy", losses)
