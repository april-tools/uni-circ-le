import functools
import argparse
import time
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "cirkit"))
sys.path.append(os.path.join(os.getcwd(), "src"))
print = functools.partial(print, flush=True)

import numpy as np
import torch

from real_qt import RealQuadTree
from trees import TREE_DICT
from clt import tree2rg
from reparam import ReparamReLU, ReparamSoftplus
from utils import num_of_params, get_date_time_str
from torch.utils.tensorboard import SummaryWriter

# cirkit
from cirkit.models.tensorized_circuit import TensorizedPC
from cirkit.models.functional import integrate
from cirkit.reparams.leaf import ReparamExp, ReparamIdentity, ReparamLeaf, ReparamSoftmax
from cirkit.layers.input.exp_family.categorical import CategoricalLayer
from cirkit.layers.input.exp_family.binomial import BinomialLayer
from cirkit.layers.sum_product import CollapsedCPLayer, TuckerLayer, SharedCPLayer
from cirkit.region_graph import RegionGraph
from cirkit.region_graph.poon_domingos import PoonDomingos
from cirkit.region_graph.quad_tree import QuadTree


parser = argparse.ArgumentParser()
parser.add_argument("--gpu",        type=int,                               help="Which gpu to use")
parser.add_argument("--log-dir",    type=str,       default="benchmark",    help="Base dir for saving the model")
parser.set_defaults(train_mode=True)
parser.add_argument('--train-mode', dest='train',   action='store_true',    help='benchmark in training mode')
parser.add_argument('--test-mode',  dest='train',   action='store_false',   help='benchmark in test mode')
parser.add_argument("--dataset",    type=str,       default="mnist",        help="Dataset for the experiment")
parser.add_argument("--num-steps",  type=int,       default=500,            help="num steps over which averaging")
parser.add_argument("--batch-size", type=int,       default=128,            help="batch_size")
parser.add_argument("--rg",         type=str,       default="QT",           help="Region graph: 'PD', 'QG', 'QT' or 'RQT'")
parser.add_argument("--input-type", type=str,       default="cat",          help="input type: either 'cat' or 'bin'")
parser.add_argument("--layer",      type=str,                               help="Layer type: 'tucker', 'cp' or 'cp-shared'")
parser.add_argument("--reparam",    type=str,       default="clamp",        help="Either 'exp', 'relu', 'exp_temp' or 'clamp'")
parser.add_argument("--k",          type=int,       default=128,            help="Num categories for mixtures")
parser.add_argument("--k-in",       type=int,       default=None,           help="Num input distributions per input region, if None then is equal to k")
args = parser.parse_args()
device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu"
if args.k_in is None: args.k_in = args.k
print(args)

#######################################################################################
################################## instantiate model ##################################
#######################################################################################

LAYER_TYPES = {
    "tucker": TuckerLayer,
    "cp": CollapsedCPLayer,
    "cp-shared": SharedCPLayer,
}
INPUT_TYPES = {"cat": CategoricalLayer, "bin": BinomialLayer}
REGION_GRAPHS = {
    'QG': QuadTree(width=28, height=28, struct_decomp=False),
    'QT': QuadTree(width=28, height=28, struct_decomp=True),
    'PD': PoonDomingos(shape=(28, 28), delta=4),
    'CLT': tree2rg(TREE_DICT[args.dataset]),
    'RQT': RealQuadTree(width=28, height=28)
}
REPARAM_TYPES = {
    "exp": ReparamExp,
    "relu": ReparamReLU,
    "softplus": ReparamSoftplus,
    "clamp": ReparamIdentity,
    "softmax": ReparamSoftmax
}

efamily_kwargs: dict = {
    'cat': {'num_categories': 256},
    'bin': {'n': 256}
}[args.input_type]

pc = TensorizedPC.from_region_graph(
    rg=REGION_GRAPHS[args.rg],
    layer_cls=LAYER_TYPES[args.layer],
    efamily_cls=INPUT_TYPES[args.input_type],
    efamily_kwargs=efamily_kwargs,
    num_inner_units=args.k,
    num_input_units=args.k_in,
    reparam=REPARAM_TYPES[args.reparam],
).to(device)
print(f"Num of params: {num_of_params(pc)}")

sqrt_eps = np.sqrt(torch.finfo(torch.get_default_dtype()).tiny)  # todo find better place
pc_pf: TensorizedPC = integrate(pc)

########################################################################################################
################################## evaluate time & space requirements ##################################
########################################################################################################

batch = torch.randint(256, (args.batch_size, pc.num_vars, 1), dtype=torch.float32).to(device)

time_per_batch = []
if args.train_mode:
    optimizer = torch.optim.Adam(pc.parameters())  # just keep everything default
    for _ in range(args.num_steps):
        tik = time.time()
        optimizer.zero_grad()
        log_likelihood = (pc(batch) - pc_pf(batch)).sum(dim=0)
        (-log_likelihood).backward()
        optimizer.step()
        if args.reparam == "clamp":
            for layer in pc.inner_layers:
                if type(layer) in [CollapsedCPLayer, SharedCPLayer]:  # note, those are collapsed but we should also include non collapsed versions
                    layer.params_in().data.clamp_(min=sqrt_eps)
                else:
                    layer.params().data.clamp_(min=sqrt_eps)
        time_per_batch.append(time.time() - tik)
else:
    if args.reparam == "clamp":
        for layer in pc.inner_layers:
            if type(layer) in [CollapsedCPLayer, SharedCPLayer]:  # note, those are collapsed but we should also include non collapsed versions
                layer.params_in().data.clamp_(min=sqrt_eps)
            else:
                layer.params().data.clamp_(min=sqrt_eps)
    for _ in range(args.num_steps):
        tik = time.time()
        with torch.no_grad():
            (pc(batch) - pc_pf(batch)).sum(dim=0);  # semicolon avoids printing
        time_per_batch.append(time.time() - tik)

time_per_batch = time_per_batch[1:]
mu_t, sigma_t = np.mean(time_per_batch), np.std(time_per_batch)
print(f"Time (ms): {mu_t:.3f}+-{sigma_t:.3f}")
if device != "cpu":
    gpu_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
    gpu_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
    print(f"GPU memory allocated (GiB): {gpu_allocated:.3f}")
    print(f"CPU memory reserved (GiB): {gpu_allocated:.3f}")

#######################################################################################################
######################################### save results ################################################
#######################################################################################################

writer = SummaryWriter(log_dir=os.path.join(args.log_dir, get_date_time_str()))
writer.add_hparams(
    hparam_dict=vars(args),
    metric_dict={
        'num_params':       num_of_params(pc),
        'time_avg':         mu_t,
        'time_std':         sigma_t,
        'gpu_allocated':    gpu_allocated if device != "cpu" else 0,
        'gpu_reserved':     gpu_reserved if device != "cpu" else 0
    },
    hparam_domain_discrete={
        'dataset':      ['mnist', 'fashion_mnist', 'celeba'],
        'rg':           [rg_name for rg_name in REGION_GRAPHS],
        'layer':        [layer for layer in LAYER_TYPES],
        'input_type':   [input_type for input_type in INPUT_TYPES],
        'reparam':      [reparam for reparam in REPARAM_TYPES]
    },
)
writer.close()
