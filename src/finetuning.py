import sys
import os
# sys.path.append(os.path.join(os.getcwd(), "src"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ten-pcs/')))

from typing import Literal
import functools
print = functools.partial(print, flush=True)

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import torch
import time

from utils import check_validity_params, init_random_seeds, get_date_time_str, count_trainable_parameters
import data.datasets as datasets
from measures import eval_loglikelihood_batched, ll2bpd, eval_bpd


from tenpcs.models.tensorized_circuit import TensorizedPC
from tenpcs.models.functional import integrate
from tenpcs.layers.sum_product import CollapsedCPLayer, SharedCPLayer, UncollapsedCPLayer
from tenpcs.layers.cp_shared import ScaledSharedCPLayer


parser = argparse.ArgumentParser()
parser.add_argument("--seed",           type=int,   default=42,         help="Random seed")
parser.add_argument("--gpu",            type=int,   default=0,          help="Device on which run the benchmark")
parser.add_argument("--dataset",        type=str,   default="mnist",    help="Dataset for the experiment")
parser.add_argument("--model-path",     type=str,                       help="Path of the model to finetune")
parser.add_argument("--lr",             type=float, default=0.1,        help="learning rate")
parser.add_argument("--rg",             type=str,                       help="'QG' or 'PD'")
parser.add_argument("--rank",           type=str,                       help="Rank")
parser.add_argument("--patience",       type=int,   default=5,          help='patience for early stopping')
parser.add_argument("--weight-decay",   type=float, default=0,          help="Weight decay coefficient")
parser.add_argument("--max-num-epochs", type=int,   default=None,       help="Max num epoch")
parser.add_argument("--batch-size",     type=int,   default=128,        help="batch size")
parser.add_argument("--progressbar",    type=bool,  default=False,      help="Print the progress bar")
parser.add_argument('--valid_freq',     type=int,   default=None,       help='validation every n steps')
parser.add_argument("--t0",             type=int,   default=1,          help='sched CAWR t0, 1 for fixed lr ')
parser.add_argument("--eta-min",        type=float, default=1e-4,       help='sched CAWR eta min')
parser.add_argument("--num-workers",    type=int,   default=0,          help="Num workers for data loader")


args = parser.parse_args()
print(args)
init_random_seeds(seed=args.seed)

device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu"

###########################################################################
################### load dataset & create logging utils ###################
###########################################################################

train, valid, test = datasets.load_dataset(args.dataset, root="../data/")
image_size = int(np.sqrt(train[0].shape[0]))  # assumes squared images
num_channels = train[0].shape[1]

train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)


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


base_name: str = os.path.splitext(os.path.basename(args.model_path))[0]
save_model_path: str = os.path.join(os.path.dirname(args.model_path), base_name + "finetuned.mdl")
writer = SummaryWriter(log_dir=os.path.dirname(save_model_path))

#######################################################################################
################################## instantiate model ##################################
#######################################################################################
pc: TensorizedPC = torch.load(args.model_path).to(device)
print(pc)
print(f"Num of params: {count_trainable_parameters(pc)}")

sqrt_eps = np.sqrt(torch.finfo(torch.get_default_dtype()).tiny)  # todo find better place
pc_pf: TensorizedPC = integrate(pc)
torch.set_default_tensor_type("torch.FloatTensor")

#######################################################################################
################################ eval model pre-finetuning ############################
#######################################################################################

compression_valid_bpd = eval_bpd(pc, valid_loader, device)
compression_test_bpd = eval_bpd(pc, test_loader, device)

print('Pre-finetuning')
print('valid bpd: ', compression_valid_bpd)
print('test  bpd: ', compression_test_bpd)

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

# note, do this before starting
for layer in pc.inner_layers:
    # note, those are collapsed but we should also include non collapsed versions
    if type(layer) in [UncollapsedCPLayer, CollapsedCPLayer, ScaledSharedCPLayer, SharedCPLayer]:
        layer.params_in().data.clamp_(min=sqrt_eps)
        if isinstance(layer, ScaledSharedCPLayer):
            layer.params_scale().data.clamp_(min=sqrt_eps)
        if isinstance(layer, UncollapsedCPLayer):
            layer.params_out().data.clamp_(min=sqrt_eps)
    else:
        layer.params().data.clamp_(min=sqrt_eps)

for epoch_count in range(1, args.max_num_epochs + 1):

    if args.valid_freq is None:
        pbar = train_loader
    else:
        pbar = DataLoader(train[torch.randint(len(train), size=(args.valid_freq * args.batch_size, ))], batch_size=args.batch_size)
    if args.progressbar:
        pbar = tqdm(iterable=pbar, total=len(pbar), unit="steps", ascii=" ▖▘▝▗▚▞█", ncols=120)

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
        for layer in pc.inner_layers:
            # note, those are collapsed but we should also include non collapsed versions
            if type(layer) in [UncollapsedCPLayer, CollapsedCPLayer, ScaledSharedCPLayer, SharedCPLayer]:
                layer.params_in().data.clamp_(min=sqrt_eps)
                if isinstance(layer, ScaledSharedCPLayer):
                    layer.params_scale().data.clamp_(min=sqrt_eps)
                if isinstance(layer, UncollapsedCPLayer):
                    layer.params_out().data.clamp_(min=sqrt_eps)
            else:
                layer.params().data.clamp_(min=sqrt_eps)

        # if args.progressbar and batch_count % 10 == 0:
        #     pbar.set_description(f"Epoch {epoch_count} Train LL={log_likelihood.item() / args.batch_size :.2f})")

    train_ll = train_ll / len(train_loader.dataset)
    valid_ll = eval_loglikelihood_batched(pc, valid_loader, device=device)

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

        print(f"[{epoch_count}-th valid step] Train bpd {ll2bpd(train_ll, pc.num_vars * pc.input_layer.num_channels):.5f}, "
              f"Valid bpd {ll2bpd(valid_ll, pc.num_vars * pc.input_layer.num_channels):.5f}, Best valid LL {best_valid_ll:.5f}")
        if device != "cpu":
            print('max allocated GPU: %.2f' % (torch.cuda.max_memory_allocated(device=device) / 1024 ** 3))

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
    hparam_dict={
        "k": pc.input_layer.num_output_units,
        "rank": args.rank,
        "rg": args.rg,
        "dataset": args.dataset
    },
    metric_dict={
        'Compression/Valid/bpd': float(compression_valid_bpd),
        'Compression/Test/bpd': float(compression_test_bpd),
        'Best/Valid/ll':    float(best_valid_ll),
        'Best/Valid/bpd':   float(ll2bpd(best_valid_ll, pc.num_vars * pc.input_layer.num_channels)),
        'Best/Test/ll':     float(best_test_ll),
        'Best/Test/bpd':    float(ll2bpd(best_test_ll, pc.num_vars * pc.input_layer.num_channels)),
        'train_time':       float(train_time),
        'num_params':       count_trainable_parameters(pc)
    },
    hparam_domain_discrete={
        'dataset':      ["celeba"] + [dataset for dataset in datasets.MNIST_NAMES],
        'rg':           ['QG', 'PD'],
    },
)
writer.close()
