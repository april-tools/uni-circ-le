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

from cirkit_extension.trees import TREE_DICT
from clt import tree2rg
from pic import PIC, zw_quadrature, parameterize_pc
from utils import check_validity_params, init_random_seeds, get_date_time_str, count_parameters, count_pc_params, \
    param_to_buffer
from datasets import load_dataset
from measures import eval_loglikelihood_batched, ll2bpd

# cirkit
from cirkit_extension.tensorized_circuit import TensorizedPC
from cirkit.models.functional import integrate
from cirkit.layers.input.exp_family.categorical import CategoricalLayer
from cirkit.layers.input.exp_family.binomial import BinomialLayer
from cirkit.layers.sum_product import CollapsedCPLayer, TuckerLayer, SharedCPLayer
from cirkit.region_graph.poon_domingos import PoonDomingos
from cirkit.region_graph.quad_tree import QuadTree
from cirkit_extension.real_qt import RealQuadTree

parser = argparse.ArgumentParser()
parser.add_argument("--seed",           type=int,   default=42,         help="Random seed")
parser.add_argument("--gpu",            type=int,   default=0,          help="Device on which run the benchmark")
parser.add_argument("--dataset",        type=str,   default="mnist",    help="Dataset for the experiment")
parser.add_argument("--model-dir",      type=str,   default="out",      help="Base dir for saving the model")
parser.add_argument("--lr",             type=float, default=0.1,        help="Path of the model to be loaded")
parser.add_argument("--patience",       type=int,   default=5,          help='patience for early stopping')
parser.add_argument("--weight-decay",   type=float, default=0,          help="Weight decay coefficient")
parser.add_argument("--k",              type=int,   default=128,        help="Num categories for mixtures")
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

assert args.layer in LAYER_TYPES
assert args.input_type in INPUT_TYPES
device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu"

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
    num_input_units=args.k,
    num_channels=num_channels
).to(device)
param_to_buffer(pc)

matrices_per_layer = []
for layer in pc.inner_layers:
    matrices_per_layer.append(layer.params_in.param.size()[:2].numel())

pic = PIC(
    n_inner_layers=np.sum(matrices_per_layer),
    inner_layer_type='cp',
    n_input_layers=pc.num_vars,
    input_layer_type='categorical',
    n_categories=256,
    single_input_net=True,
    multi_heads_inner_net=True
).to(device)

print(f"QPC num of params: {count_pc_params(pc)}")
print(f"PIC num of params: {count_parameters(pic)}")

#######################################################################################
################################ optimizer & scheduler ################################
#######################################################################################

optimizer = torch.optim.Adam(pic.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=1, eta_min=args.eta_min)

###############################################################################
################################ training loop ################################
###############################################################################

# steps = number of integration points = K
z, log_w = zw_quadrature('trapezoidal', nip=args.k, a=-1, b=1, device=device)

best_valid_ll = -np.infty
patience_counter = args.patience

tik_train = time.time()
for epoch_count in range(1, args.max_num_epochs + 1):

    if args.valid_freq is None:
        pbar = train_loader
    else:
        pbar = DataLoader(train[torch.randint(len(train), size=(args.valid_freq * args.batch_size,))],
                          batch_size=args.batch_size)
    if args.progressbar:
        pbar = tqdm(iterable=pbar, total=len(pbar), unit="steps", ascii=" ▖▘▝▗▚▞█", ncols=120)

    train_ll = 0
    for batch_count, batch in enumerate(pbar):
        sum_param, input_param = pic.quad(z=z, log_w=log_w)
        parameterize_pc(pc, sum_param, input_param)

        batch = batch.to(device)  # (batch_size, num_vars, num channels)
        log_likelihood = pc(batch).sum(dim=0)
        optimizer.zero_grad()
        (-log_likelihood).backward()
        train_ll += log_likelihood.item()
        check_validity_params(pc)
        optimizer.step()
        scheduler.step()
        check_validity_params(pc)

    train_ll = train_ll / len(train_loader.dataset)
    valid_ll = eval_loglikelihood_batched(pc, valid_loader, device=device)

    print(f"[{epoch_count}-th valid step]", 'train LL %.5f, valid LL %.5f, best valid LL %.5f' % (train_ll, valid_ll, best_valid_ll))
    if device != "cpu": print('max allocated GPU: %.2f' % (torch.cuda.max_memory_allocated() / 1024 ** 3))

    # Not improved
    if valid_ll <= best_valid_ll:
        patience_counter -= 1
        if patience_counter == 0:
            print("-> Validation LL did not improve, early stopping")
            break
    else:
        print("-> Saved model")
        torch.save(pic, save_model_path)
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

pic = torch.load(save_model_path)
sum_param, leaf_param = pic.quad(z=z, log_w=log_w)
parameterize_pc(pc, sum_param, input_param)

pc_pf = integrate(pc)
print('norm const', pc_pf(None))

best_train_ll = eval_loglikelihood_batched(pc, train_loader, device=device)
best_test_ll = eval_loglikelihood_batched(pc, test_loader, device=device)

print('train bpd: ', ll2bpd(best_train_ll, pc.num_vars * num_channels))
print('valid bpd: ', ll2bpd(best_valid_ll, pc.num_vars * num_channels))
print('test  bpd: ', ll2bpd(best_test_ll, pc.num_vars * num_channels))

writer.add_hparams(
    hparam_dict=vars(args),
    metric_dict={
        'Best/Valid/ll':    float(best_valid_ll),
        'Best/Valid/bpd':   float(ll2bpd(best_valid_ll, pc.num_vars)),
        'Best/Test/ll':     float(best_test_ll),
        'Best/Test/bpd':    float(ll2bpd(best_test_ll, pc.num_vars)),
        'train_time':       float(train_time),
    },
    hparam_domain_discrete={
        'dataset':      ['mnist', 'fashion_mnist', 'celeba'],
        'rg':           ['QG', 'QT', 'PD', 'CLT', 'RQT'],
        'layer':        [layer for layer in LAYER_TYPES],
        'input_type':   [input_type for input_type in INPUT_TYPES]
    },
)
writer.close()
