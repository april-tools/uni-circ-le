import sys
import os
from typing import Literal
sys.path.append(os.path.join(os.getcwd(), "cirkit"))
sys.path.append(os.path.join(os.getcwd(), "src"))

import functools
print = functools.partial(print, flush=True)

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import torch
import time

from trees import TREE_DICT
from clt import tree2rg
from reparam import ReparamReLU, ReparamSoftplus
from utils import check_validity_params, init_random_seeds, get_date_time_str, num_of_params
from datasets import load_dataset
from measures import eval_loglikelihood_batched, ll2bpd


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
from real_qt import RealQuadTree


parser = argparse.ArgumentParser()
parser.add_argument("--seed",           type=int,   default=42,         help="Random seed")
parser.add_argument("--gpu",            type=int,   default=0,          help="Device on which run the benchmark")
parser.add_argument("--dataset",        type=str,   default="mnist",    help="Dataset for the experiment")
parser.add_argument("--model-dir",      type=str,   default="out",      help="Base dir for saving the model")
parser.add_argument("--lr",             type=float, default=0.1,        help="Path of the model to be loaded")
parser.add_argument("--patience",       type=int,   default=5,          help='patience for early stopping')
parser.add_argument("--weight-decay",   type=float, default=0,          help="Weight decay coefficient")
parser.add_argument("--k",              type=int,   default=128,        help="Num categories for mixtures")
parser.add_argument("--k-in",           type=int,   default=None,       help="Num input distributions per input region, if None then is equal to k",)
parser.add_argument("--rg",             type=str,   default="QT",       help="Region graph: 'PD', 'QG', 'QT' or 'RQT'")
parser.add_argument("--layer",          type=str,                       help="Layer type: 'tucker', 'cp' or 'cp-shared'")
parser.add_argument("--input-type",     type=str,                       help="input type: either 'cat' or 'bin'")
parser.add_argument("--reparam",        type=str,   default="clamp",    help="Either 'exp', 'relu', 'exp_temp' or 'clamp'")
parser.add_argument("--max-num-epochs", type=int,   default=None,       help="Max num epoch")
parser.add_argument("--batch-size",     type=int,   default=128,        help="batch size")
parser.add_argument("--progressbar",    type=bool,  default=False,      help="Print the progress bar")
parser.add_argument('--valid_freq',     type=int,   default=None,       help='validation every n steps')
parser.add_argument("--t0",             type=int,   default=1,          help='sched CAWR t0, 1 for fixed lr ')
parser.add_argument("--eta-min",        type=float, default=1e-4,       help='sched CAWR eta min')
args = parser.parse_args()
print(args)
init_random_seeds(seed=args.seed)


LAYER_TYPES = {
    "tucker": TuckerLayer,
    "cp": CollapsedCPLayer,
    "cp-shared": SharedCPLayer,
}
INPUT_TYPES = {"cat": CategoricalLayer, "bin": BinomialLayer}
REPARAM_TYPES = {
    "exp": ReparamExp,
    "relu": ReparamReLU,
    "softplus": ReparamSoftplus,
    "clamp": ReparamIdentity,
    "softmax": ReparamSoftmax
    # "exp_temp" will be added at run-time
    # "softmax_temp" will be added at run-time
}

assert args.layer in LAYER_TYPES
assert args.input_type in INPUT_TYPES
device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu"
if args.k_in is None: args.k_in = args.k

###########################################################################
################### load dataset & create logging utils ###################
###########################################################################

train, valid, test = load_dataset(args.dataset)
image_size = int(np.sqrt(train[0].shape[0]))  # assumes squared images
num_channels = train[0].shape[1]

train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)


def make_path(base_dir, intermediate_dir: Literal["models", "logs"]):
    return os.path.join(
        base_dir,
        intermediate_dir,
        args.dataset,
        args.rg,
        args.layer,
        args.input_type,
        args.reparam,
        f"k_{args.k}",
        f"lr_{args.lr}",
        f"b_{args.batch_size}",
        get_date_time_str() + ".mdl")


save_model_path: str = make_path(args.model_dir, "models")
save_log_path: str = make_path(args.model_dir, "logs")
model_id: str = os.path.splitext(os.path.basename(save_model_path))[0]
writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(save_log_path), model_id))
if not os.path.exists(os.path.dirname(save_model_path)): os.makedirs(os.path.dirname(save_model_path))

#######################################################################################
################################## instantiate model ##################################
#######################################################################################

if args.rg == 'QG':
    rg = QuadTree(width=image_size, height=image_size, struct_decomp=False)
elif args.rg == 'QT':
    rg = QuadTree(width=image_size, height=image_size, struct_decomp=True)
elif args.rg == 'PD':
    rg = PoonDomingos(shape=(image_size, image_size), delta=4)
elif args.rg == 'CLT':
    rg = tree2rg(TREE_DICT[args.dataset])
elif args.rg == 'RQT':
    rg = RealQuadTree(width=image_size, height=image_size)
else:
    raise NotImplementedError("region graph not available")

efamily_kwargs: dict = {
    'cat': {'num_categories': 256},
    'bin': {'n': 256}
}[args.input_type]

pc = TensorizedPC.from_region_graph(
    rg=rg,
    layer_cls=LAYER_TYPES[args.layer],
    efamily_cls=INPUT_TYPES[args.input_type],
    efamily_kwargs=efamily_kwargs,
    num_inner_units=args.k,
    num_input_units=args.k_in,
    num_channels=num_channels,
    reparam=REPARAM_TYPES[args.reparam],
).to(device)
print(f"Num of params: {num_of_params(pc)}")

sqrt_eps = np.sqrt(torch.finfo(torch.get_default_dtype()).tiny)  # todo find better place
pc_pf: TensorizedPC = integrate(pc)
torch.set_default_tensor_type("torch.FloatTensor")

#######################################################################################
################################ optimizer & scheduler ################################
#######################################################################################

optimizer = torch.optim.Adam([
    {'params': [p for p in pc.input_layer.parameters()]},
    {'params': [p for layer in pc.inner_layers for p in layer.parameters()], 'weight_decay': args.weight_decay}], lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=1, eta_min=args.eta_min)

###############################################################################
################################ training loop ################################
###############################################################################

best_valid_ll = -np.infty
patience_counter = args.patience

tik_train = time.time()
for epoch_count in range(1, args.max_num_epochs + 1):

    if args.valid_freq is None:
        pbar = train_loader
    else:
        pbar = DataLoader(train[torch.randint(len(train), size=(args.valid_freq * args.batch_size, ))], batch_size=args.batch_size)
    if args.progressbar: pbar = tqdm(iterable=pbar, total=len(pbar), unit="steps", ascii=" ▖▘▝▗▚▞█", ncols=120)

    train_ll = 0
    for batch_count, batch in enumerate(pbar):

        batch = batch.to(device)  # (batch_size, num_vars, num channels)
        log_likelihood = (pc(batch) - pc_pf(batch)).sum(dim=0)
        optimizer.zero_grad()
        (-log_likelihood).backward()
        train_ll += log_likelihood.item()
        check_validity_params(pc)
        optimizer.step()
        scheduler.step()
        check_validity_params(pc)

        # project params in inner layers TODO: remove or edit?
        if args.reparam == "clamp":
            for layer in pc.inner_layers:
                if type(layer) in [CollapsedCPLayer, SharedCPLayer]: # note, those are collapsed but we should also include non collapsed versions
                    layer.params_in().data.clamp_(min=sqrt_eps)
                else:
                    layer.params().data.clamp_(min=sqrt_eps)

        # if args.progressbar and batch_count % 10 == 0:
        #     pbar.set_description(f"Epoch {epoch_count} Train LL={log_likelihood.item() / args.batch_size :.2f})")

    train_ll = train_ll / len(train_loader.dataset)
    valid_ll = eval_loglikelihood_batched(pc, valid_loader, device=device)

    print(f"[{epoch_count}-th valid step]", 'train LL %.2f, valid LL %.2f' % (train_ll, valid_ll))
    if device != "cpu": print('max allocated GPU: %.2f' % (torch.cuda.max_memory_allocated() / 1024 ** 3))

    # Not improved
    if valid_ll <= best_valid_ll:
        patience_counter -= 1
        if patience_counter == 0:
            print("-> Validation LL did not improve, early stopping")
            break
    else:
        print("-> Saved model")
        torch.save(pc, save_model_path)
        best_valid_ll = valid_ll
        patience_counter = args.patience

    writer.add_scalar("train_ll", train_ll, epoch_count)
    writer.add_scalar("valid_ll", valid_ll, epoch_count)
    writer.flush()

train_time = time.time() - tik_train
print(f'Overall training time: {train_time:.2f} (s)')

#########################################################################
################################ testing ################################
#########################################################################

pc: TensorizedPC = torch.load(save_model_path)
best_train_ll = eval_loglikelihood_batched(pc, train_loader, device=device)
best_test_ll = eval_loglikelihood_batched(pc, test_loader, device=device)

print('train bpd: ', ll2bpd(best_train_ll, pc.num_vars * pc.input_layer.num_channels))
print('valid bpd: ', ll2bpd(best_valid_ll, pc.num_vars * pc.input_layer.num_channels))
print('test  bpd: ', ll2bpd(best_test_ll, pc.num_vars * pc.input_layer.num_channels))


writer.add_hparams(
    hparam_dict=vars(args),
    metric_dict={
        'Best/Valid/ll':    float(best_valid_ll),
        'Best/Valid/bpd':   float(ll2bpd(best_valid_ll, pc.num_vars * pc.input_layer.num_channels)),
        'Best/Test/ll':     float(best_test_ll),
        'Best/Test/bpd':    float(ll2bpd(best_test_ll, pc.num_vars * pc.input_layer.num_channels)),
        'train_time':       float(train_time),
    },
    hparam_domain_discrete={
        'dataset':      ['mnist', 'fashion_mnist', 'celeba'],
        'rg':           ['QG', 'QT', 'PD', 'CLT', 'RQT'],
        'layer':        [layer for layer in LAYER_TYPES],
        'input_type':   [input_type for input_type in INPUT_TYPES],
        'reparam':      [reparam for reparam in REPARAM_TYPES]
    },
)
writer.close()
