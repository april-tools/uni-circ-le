import sys
import os
sys.path.append(os.path.join(os.getcwd(), "cirkit"))
sys.path.append(os.path.join(os.getcwd(), "src"))

import functools
print = functools.partial(print, flush=True)

from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from tqdm import tqdm
import numpy as np
import argparse
import torch
import time

from reparam import ReparamReLU, ReparamSoftplus
from utils import load_dataset, check_validity_params, init_random_seeds, get_date_time_str, num_of_params
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
parser.add_argument("--rg",             type=str,   default="QT",       help="Region graph: 'PD', 'QG', or 'QT'")
parser.add_argument("--layer",          type=str,                       help="Layer type: 'tucker', 'cp' or 'cp-shared'")
parser.add_argument("--input-type",     type=str,                       help="input type: either 'cat' or 'bin'")
parser.add_argument("--reparam",        type=str,   default="exp",      help="Either 'exp', 'relu', or 'exp_temp'")
parser.add_argument("--max-num-epochs", type=int,   default=200,        help="Max num epoch")
parser.add_argument("--batch-size",     type=int,   default=128,        help="batch size")
parser.add_argument("--progressbar",    type=bool,  default=False,      help="Print the progress bar")
parser.add_argument('--t0',              type=int,   default=1,          help='sched CAWR t0, 1 for fixed lr ')
parser.add_argument('--eta-min',         type=float, default=1e-4,       help='sched CAWR eta min')
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
assert args.rg in ['QG', 'QT', 'PD', 'RND', 'CLT']
assert args.input_type in INPUT_TYPES
device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu"
if args.k_in is None: args.k_in = args.k


###########################################################################
################### load dataset & create logging utils ###################
###########################################################################

train, valid, test = load_dataset(args.dataset, device='cpu')

save_path = os.path.join(
    args.model_dir,
    args.dataset,
    args.rg,
    args.layer,
    args.input_type,
    args.reparam,
    f"k_{args.k}",
    f"lr_{args.lr}",
    f"b_{args.batch_size}",
    get_date_time_str() + ".mdl",
)
model_id: str = os.path.splitext(os.path.basename(save_path))[0]
writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(save_path), model_id))
if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path))

#######################################################################################
################################## instantiate model ##################################
#######################################################################################

rg: RegionGraph = {
    'QG': QuadTree(width=28, height=28, struct_decomp=False),
    'QT': QuadTree(width=28, height=28, struct_decomp=True),
    'PD': PoonDomingos(shape=(28, 28), delta=4)
}[args.rg]

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

best_valid_ll = eval_loglikelihood_batched(pc, valid, device=device) / valid.shape[0]
patience_counter = args.patience

tik_train = time.time()
for epoch_count in range(1, args.max_num_epochs + 1):

    idx_batches = torch.randperm(train.shape[0]).split(args.batch_size)
    pbar = enumerate(idx_batches)
    if args.progressbar: pbar = tqdm(iterable=pbar, total=len(idx_batches), unit="steps", ascii=" ▖▘▝▗▚▞█", ncols=120)

    train_ll = 0
    for batch_count, idx in pbar:
        batch_x = train[idx, :].unsqueeze(dim=-1).to(device)
        log_likelihood = (pc(batch_x) - pc_pf(batch_x)).sum(dim=0)

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
                    layer.params_in().data = torch.clamp(layer.params_in(), min=sqrt_eps)
                else:
                    layer.params().data = torch.clamp(layer.params(), min=sqrt_eps)

        if args.progressbar and batch_count % 10 == 0:
            pbar.set_description(f"Epoch {epoch_count} Train LL={log_likelihood.item() / args.batch_size :.2f})")

    train_ll = train_ll / train.shape[0]
    valid_ll = eval_loglikelihood_batched(pc, valid, device=device) / valid.shape[0]

    print(f"[After epoch {epoch_count}]", 'train LL %.2f, valid LL %.2f' % (train_ll, valid_ll))
    if device != "cpu": print('Max allocated GPU: %.2f' % (torch.cuda.max_memory_allocated() / 1024 ** 3))

    # Not improved
    if valid_ll <= best_valid_ll:
        patience_counter -= 1
        if patience_counter == 0:
            print("-> Validation LL did not improve, early stopping")
            break
    else:
        print("-> Saved model")
        torch.save(pc, save_path)
        best_valid_ll = valid_ll
        patience_counter = args.patience

    writer.add_scalar("train_ll", train_ll, epoch_count)
    writer.add_scalar("valid_ll", valid_ll, epoch_count)
    writer.flush()

print('Overall training time: %.2f (s)' % (time.time() - tik_train))

#########################################################################
################################ testing ################################
#########################################################################

pc = torch.load(save_path)
best_train_ll = eval_loglikelihood_batched(pc, train, device=device) / train.shape[0]
best_test_ll = eval_loglikelihood_batched(pc, test, device=device) / test.shape[0]

print('train bpd: ', ll2bpd(best_train_ll, pc.num_vars))
print('valid bpd: ', ll2bpd(best_valid_ll, pc.num_vars))
print('test  bpd: ', ll2bpd(best_test_ll, pc.num_vars))

writer.add_hparams(
    hparam_dict=vars(args),
    metric_dict={
        'Best/Valid/ll':    float(best_valid_ll),
        'Best/Valid/bpd':   float(ll2bpd(best_valid_ll, pc.num_vars)),
        'Best/Test/ll':     float(best_test_ll),
        'Best/Test/bpd':    float(ll2bpd(best_test_ll, pc.num_vars)),
    },
    hparam_domain_discrete={
        'dataset':      ['mnist', 'fashion_mnist', 'celeba'],
        'rg':           ['QG', 'PD', 'QT'],
        'layer':        ['cp', 'cpshared', 'tucker'],
        'input_type':   ['bin', 'cat'],
        'reparam':      ['softplus', 'exp', 'exp_temp', 'relu', 'clamp']
    },
)
writer.close()
