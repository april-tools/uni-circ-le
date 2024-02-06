import os
import sys
import argparse
from typing import Literal
import torch

import functools
print = functools.partial(print, flush=True)

sys.path.append(os.path.join(os.getcwd(), "cirkit"))
sys.path.append(os.path.join(os.getcwd(), "src"))

from torch.utils.tensorboard import SummaryWriter
from utils import *
from measures import *
from tqdm import tqdm
import time


# cirkit
from cirkit.models.functional import integrate
from cirkit.models.tensorized_circuit import TensorizedPC
from cirkit.reparams.leaf import ReparamExp, ReparamIdentity, ReparamLeaf, ReparamSoftmax
from cirkit.layers.input.exp_family.categorical import CategoricalLayer
from cirkit.layers.input.exp_family.binomial import BinomialLayer
from cirkit.layers.sum_product import CollapsedCPLayer, TuckerLayer, SharedCPLayer
from cirkit.models.tensorized_circuit import TensorizedPC
from cirkit.region_graph import RegionGraph
from cirkit.region_graph.poon_domingos import PoonDomingos
from cirkit.region_graph.quad_tree import QuadTree


parser = argparse.ArgumentParser("MNIST experiments.")
parser.add_argument("--gpu", type=int, default=None, help="Device on which run the benchmark")
parser.add_argument("--dataset", type=str, default="mnist", help="Dataset for the experiment")
parser.add_argument("--model_path", type=str, default="out", help="complete path where the model is stored")
args = parser.parse_args()
print(args)

device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu"
pc = torch.load(args.model_path).to(device)

x_train, x_valid, x_test = load_dataset(args.dataset, device="cpu")

print('train bpd: ', eval_bpd(pc, x_train, device=device))
print('valid bpd: ', eval_bpd(pc, x_valid, device=device))
print('test  bpd: ', eval_bpd(pc, x_test, device=device))
