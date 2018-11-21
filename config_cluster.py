import json
import pathlib
import os

model = "causal"
seg_length = 20000
lam = float(os.environ['lam'])
lr = float(os.environ['lr'])
change_lr = 5000
bs = 2
ks = 250
optim = "SGD"
nthresh = 0.01
bpti = True
thresh = 0.1

# BASE_DIR = "/mnt/c/Users/Eric/Documents/Berkeley/Research/Neuroscience/Sparse coding/"
BASE_DIR = "/global/home/users/edodds/"
EXP_DIR = BASE_DIR + "convsparse/Experiments/" #/home/edodds/convsparse/Experiments/"
EXP_SUBDIR = EXP_DIR + "{}timit_lam{}_lr{}_ks{}-nt{}-000".format(model,
                                                                 lam,
                                                                 lr,
                                                                 ks,
                                                                 nthresh)

config = {}

if change_lr > 0:
    schedule = {"steps": [0, change_lr], "rates": [lr, lr/10]}
else:
    schedule = {"steps": [0], "rates": [lr]}

DATA_DIR = BASE_DIR + "audition/Data/TIMIT/"
config.update({"data_folder": DATA_DIR,#"/home/edodds/Data/TIMIT/",
               "experiment_folder": EXP_SUBDIR,
               "model": model,
               "segment_length": seg_length,
               "sparseness_parameter": lam,
               "learning_schedule": schedule,
               "optimizer": optim,
               "batch_size": bs,
               "kernel_size": ks,
               "signal_normalization": 0,
               "divide_out_signal_power": True,
               "normed_thresh": nthresh,
               "thresh": thresh,
               "backprop_through_inference": bpti})

pathlib.Path(EXP_SUBDIR).mkdir(parents=True, exist_ok=True)
json_file = EXP_SUBDIR+"/config.json"
with open(json_file, 'w') as fh:
    json.dump(config, fh)

print("Saved configuration to:\n{}".format(json_file))
