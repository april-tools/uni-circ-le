import os
import sys
import argparse

import pandas as pd

import functools

from datasets import load_dataset

print = functools.partial(print, flush=True)

sys.path.append(os.path.join(os.getcwd(), "cirkit"))
sys.path.append(os.path.join(os.getcwd(), "src"))

from utils import *
from measures import *
from cirkit_extension.tensorized_circuit import TensorizedPC


parser = argparse.ArgumentParser("MNIST experiments.")
parser.add_argument("--gpu", type=int, default=None, help="Device on which run the benchmark")
parser.add_argument("--dataset", type=str, default="mnist", help="Dataset for the experiment")
parser.add_argument("--model_path", type=str, default="out", help="complete path where the model is stored")
parser.add_argument("--save-results", type=str, help="path of the csv where to save results")
args = parser.parse_args()
print(args)

device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu"
pc: TensorizedPC = torch.load(args.model_path).to(device)

train, valid, test = load_dataset(args.dataset)
image_size = int(np.sqrt(train[0].shape[0]))  # assumes squared images
num_channels = train[0].shape[1]

train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

train_bpd = eval_bpd(pc, train_loader, device=device)
valid_bpd = eval_bpd(pc, valid_loader, device=device)
test_bpd = eval_bpd(pc, test_loader, device=device)
print('train bpd: ', train_bpd)
print('valid bpd: ', valid_bpd)
print('test  bpd: ', test_bpd)

csv_row = {
    "path": args.model_path,
    "train_bpd": train_bpd,
    "valid_bpd": valid_bpd,
    "test_bpd": test_bpd
}

df = pd.DataFrame.from_dict([csv_row])
if os.path.exists(args.results_csv):
    df.to_csv(args.results_csv, mode="a", index=False, header=False)
else:
    df.to_csv(args.results_csv, index=False)
