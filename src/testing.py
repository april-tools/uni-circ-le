import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ten-pcs/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse

import functools

import data.datasets as datasets

print = functools.partial(print, flush=True)

from utils import *
from measures import *
from tenpcs.models.tensorized_circuit import TensorizedPC


parser = argparse.ArgumentParser("MNIST experiments.")
parser.add_argument("--gpu", type=int, default=None, help="Device on which run the benchmark")
parser.add_argument("--dataset", type=str, default="mnist", help="Dataset for the experiment")
parser.add_argument("--ycc", type=str, default="none", help="either 'none', 'lossless', 'lossy'")
parser.add_argument("--model_path", type=str, default="out", help="complete path where the model is stored")
parser.add_argument("--csv-file-path", type=None, help="path of the csv where to save results")
args = parser.parse_args()
print(args)

device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu"
pc: TensorizedPC = torch.load(args.model_path).to(device)

train, valid, test = datasets.load_dataset(args.dataset, ycc=args.ycc, valid_split_percentage=0.05, root='../data/')

if args.dataset == "celeba":
    num_workers = 64
else:
    num_workers = 0

train_loader = DataLoader(train, batch_size=512, shuffle=False, num_workers=num_workers)
valid_loader = DataLoader(valid, batch_size=512, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=num_workers)


train_ll = eval_loglikelihood_batched(pc, train_loader, device=device)
valid_ll = eval_loglikelihood_batched(pc, valid_loader, device=device)
test_ll = eval_loglikelihood_batched(pc, test_loader, device=device)
print('train LL: ', train_ll)
print('valid LL: ', valid_ll)
print('test  LL: ', test_ll)
train_bpd = ll2bpd(train_ll, pc.num_vars * pc.input_layer.num_channels)
valid_bpd = ll2bpd(valid_ll, pc.num_vars * pc.input_layer.num_channels)
test_bpd = ll2bpd(test_ll, pc.num_vars * pc.input_layer.num_channels)
print('train bpd: ', train_bpd)
print('valid bpd: ', valid_bpd)
print('test  bpd: ', test_bpd)

if args.csv_file_path is not None:
    import pandas as pd
    csv_row = {
        "path": args.model_path,
        "train_LL": train_ll,
        "valid_LL": valid_ll,
        "test_LL": test_ll,
        "train_bpd": train_bpd,
        "valid_bpd": valid_bpd,
        "test_bpd": test_bpd
    }

    df = pd.DataFrame.from_dict([csv_row])
    if os.path.exists(args.csv_file_path):
        df.to_csv(args.csv_file_path, mode="a", index=False, header=False)
    else:
        df.to_csv(args.csv_file_path, index=False)
