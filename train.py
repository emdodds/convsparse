import numpy as np
import torch
import argparse
import pathlib
import json

import signalset
import convsparsenet as csn
import causalMP
import matching_pursuit as mp

parser = argparse.ArgumentParser()
parser.add_argument('--c', default=None, type=str)
parser.add_argument('--device', default="cpu", type=str)
args = parser.parse_args()

from config import config
if args.c is not None:
    with open(args.c, 'r') as fh:
        specified = json.load(fh)
    for key in config:
        if key not in specified:
            print("{} not specified, defaulting to: {}".format(key, config[key]))
    config.update(specified)

data = signalset.SignalSet(data=config["data_folder"],
                           all_in_memory=False,
                           norm_factor=config["signal_normalization"])
data.seg_length = config["segment_length"]

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
                            initialization="minirandom", seed_length=800,
                            kernel_size=config["kernel_size"],
                            lam=config["sparseness_parameter"])
elif config["model"] == "mp":
    net = mp.MPNet(device=device, seed_length=100, n_iter=2000, 
                   lam=config["sparseness_parameter"],
                   initialization="minirandom",
                   kernel_size=config["kernel_size"])
else:
    raise ValueError("Unsupported model specifiction: {}".format(config["model"]))

net.batch_size = config["batch_size"]

EXP_SUBDIR = config["experiment_folder"]
pathlib.Path(EXP_SUBDIR).mkdir(parents=True, exist_ok=True)

losses = []
for tt in range(1000):
    losses = losses + net.train(data, n_steps=10,
                                learning_rate=config["learning_rate"])
    print("Saving in {}".format(EXP_SUBDIR))
    np.save(EXP_SUBDIR+"/weights.npy", np.squeeze(net.weights.detach().cpu().numpy()))
    np.save(EXP_SUBDIR+"/loss.npy", losses)
