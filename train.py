import numpy as np
import torch
import argparse
import pathlib
import json

import signalset
import convsparsenet as csn
import causalMP
import matching_pursuit as mp
from causalcsn import CausalConvSparseNet


def get_rate(schedule, step):
    ind = 0
    for ii, boundary in enumerate(schedule["steps"]):
        if boundary > step:
            break
        else:
            ind = ii
    return schedule["rates"][ind]


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
    net = csn.ConvSparseNet(n_kernel=config["n_kernel"],
                            lam=config["sparseness_parameter"],
                            initialization="minirandom", seed_length=100,
                            kernel_size=config["kernel_size"],
                            device=device)
elif config["model"] == "causalcsn":
    net = CausalConvSparseNet(n_kernel=config["n_kernel"],
                              lam=config["sparseness_parameter"],
                              initialization="minirandom",
                              seed_length=config["kernel_size"],
                              kernel_size=config["kernel_size"],
                              device=device,
                              inference_rate=config["inf_rate"],
                              n_iter=config["n_iter"])
elif config["model"] == "causal":
    net = causalMP.CausalMP(n_kernel=config["n_kernel"],
                            device=device,
                            initialization="minirandom", seed_length=800,
                            kernel_size=config["kernel_size"],
                            lam=config["sparseness_parameter"],
                            thresh=config["thresh"],
                            normed_thresh=config["normed_thresh"],
                            backprop_through_inference=config["backprop_through_inference"])
elif config["model"] == "mp":
    net = mp.MPNet(n_kernel=config["n_kernel"],
                   device=device, seed_length=100, n_iter=2000,
                   lam=config["sparseness_parameter"],
                   initialization="minirandom",
                   kernel_size=config["kernel_size"],
                   dropout=config['dropout'])
elif config["model"] == "growing":
    net = mp.Growing_MPNet(trim_threshold=config["normed_thresh"],
                           n_kernel=config["n_kernel"],
                           device=device, n_iter=2000,
                           lam=config["sparseness_parameter"],
                           initialization="minirandom",
                           kernel_size=config["kernel_size"],
                           seed_length=config["kernel_size"],
                           dropout=config['dropout'])
else:
    raise ValueError("Unsupported model specifiction: {}".format(config["model"]))

net.batch_size = config["batch_size"]

EXP_SUBDIR = config["experiment_folder"]
pathlib.Path(EXP_SUBDIR).mkdir(parents=True, exist_ok=True)

try:
    net.load(EXP_SUBDIR)
    print("Loaded from {}".format(EXP_SUBDIR))
    losses = list(np.load(EXP_SUBDIR + "/loss.npy"))
except FileNotFoundError:
    losses = []
    print("Training from scratch.")

steps_between = 10
for tt in range(1000):
    if config["stop_dropout"] < len(losses):
        try:
            net.dropout = 1.
        except AttributeError:
            pass
    losses += net.train(data, n_steps=steps_between,
                        learning_rate=get_rate(config["learning_schedule"],
                                               len(losses)),
                        optimizer=config["optimizer"],
                        step_count=len(losses),
                        divide_out_signal_power=config["divide_out_signal_power"])
    print("Saving in {}".format(EXP_SUBDIR))
    np.save(EXP_SUBDIR+"/weights.npy", np.squeeze(net.weights.detach().cpu().numpy()))
    np.save(EXP_SUBDIR+"/loss.npy", losses)
