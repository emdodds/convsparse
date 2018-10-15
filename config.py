import argparse
import json
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--load', default=None, type=str)
parser.add_argument('--seg_length', default=20000, type=int)
parser.add_argument('--lam', default=0.1, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--bs', default=4, type=int)
parser.add_argument('--ks', default=800, type=int)
parser.add_argument('--model', default="causal", type=str)
args = parser.parse_args()

EXP_DIR = "/home/edodds/convsparse/Experiments/"
EXP_SUBDIR = EXP_DIR + "{}timit_lam{}_lr{}_ks{}-001".format(args.model,
                                                            args.lam,
                                                            args.lr,
                                                            args.ks)

if args.load is not None:
    with open(args.load, 'r') as fh:
        config = json.load(fh)
else:
    config = {}

config.update({"data_folder": "/home/edodds/Data/TIMIT/",
               "experiment_folder": EXP_SUBDIR,
               "model": args.model,
               "segment_length": args.seg_length,
               "sparseness_parameter": args.lam,
               "learning_rate": args.lr,
               "batch_size": args.bs,
               "kernel_size": args.ks,
               "signal_normalization": 20}) # alt: 20

pathlib.Path(EXP_SUBDIR).mkdir(parents=True, exist_ok=True)
json_file = EXP_SUBDIR+"/config.json"
with open(json_file, 'w') as fh:
    json.dump(config, fh)

print("Saved configuration to:\n{}".format(json_file))
