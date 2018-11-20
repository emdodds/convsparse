import argparse
import json
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--load', default=None, type=str)
parser.add_argument('--seg_length', default=20000, type=int)
parser.add_argument('--lam', default=0.1, type=float)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--change_lr', default=10000, type=int)
parser.add_argument('--bs', default=4, type=int)
parser.add_argument('--ks', default=100, type=int)
parser.add_argument('--model', default="growing", type=str)
parser.add_argument('--optim', default='SGD', type=str)
parser.add_argument('--nthresh', default=0.01, type=float)
parser.add_argument('--n_kernel', default=32, type=int)
parser.add_argument('--dropout', default=0.5, type=float) # only implemented for MP
parser.add_argument('--stop_dropout', default=3000, type=int)
parser.add_argument('--inf_rate', default=1., type=float) # for causalcsn
parser.add_argument('--n_iter', default=100, type=int)
args = parser.parse_args()

# BASE_DIR = "/mnt/c/Users/Eric/Documents/Berkeley/Research/Neuroscience/Sparse coding/"
BASE_DIR = "/home/edodds/"
EXP_DIR = BASE_DIR + "convsparse/Experiments/" #/home/edodds/convsparse/Experiments/"
EXP_SUBDIR = EXP_DIR + "{}timit_lam{}_lr{}_ks{}-ni{}-nk{}-do-000".format(args.model,
                                                                      args.lam,
                                                                      args.lr,
                                                                      args.ks,
                                                                      args.n_iter,
                                                                      args.n_kernel)

if args.load is not None:
    with open(args.load, 'r') as fh:
        config = json.load(fh)
else:
    config = {}

if args.change_lr > 0:
    schedule = {"steps": [0, args.change_lr], "rates": [args.lr, args.lr/10]}
else:
    schedule = {"steps": [0], "rates": [args.lr]}

DATA_DIR = BASE_DIR + "/Data/TIMIT/"
config.update({"data_folder": DATA_DIR,#"/home/edodds/Data/TIMIT/",
               "experiment_folder": EXP_SUBDIR,
               "model": args.model,
               "segment_length": args.seg_length,
               "sparseness_parameter": args.lam,
               "learning_schedule": schedule,
               "optimizer": args.optim,
               "batch_size": args.bs,
               "kernel_size": args.ks,
               "signal_normalization": 0,
               "divide_out_signal_power": True,
               "normed_thresh": args.nthresh,
               "n_kernel": args.n_kernel,
               "dropout": args.dropout,
               "stop_dropout": args.stop_dropout,
               "inf_rate": args.inf_rate,
               "n_iter": args.n_iter})

pathlib.Path(EXP_SUBDIR).mkdir(parents=True, exist_ok=True)
json_file = EXP_SUBDIR+"/config.json"
with open(json_file, 'w') as fh:
    json.dump(config, fh)

# print("Saved configuration to:\n{}".format(json_file))

print(json_file)
