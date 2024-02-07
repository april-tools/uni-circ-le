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

from reparam import ReparamExpTemp, ReparamSoftmaxTemp, ReparamReLU, ReparamSoftplus
from utils import load_dataset, check_validity_params, init_random_seeds, get_date_time_str, num_of_params
from measures import eval_loglikelihood_batched, bpd_from_ll


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


def train_procedure(
    *,
    pc: TensorizedPC,
    pc_hypar: dict,
    dataset_name: str,
    save_path: str,
    batch_size: Optional[int] = 128,
    lr: Optional[float] = 0.01,
    weight_decay: Optional[float] = 0,
    max_num_epochs: Optional[int] = 200,
    patience: Optional[int] = 10,
    verbose: Optional[bool] = True,
    compute_train_ll: Optional[bool] = False
):
    """Train a TensorizedPC using gradient descent.

    Args:
        pc (TensorizedPC): prob. circ. to train
        dataset_name (str): dataset name
        save_path (str): path (including file name and extension) for saving the trained model
        tensorboard_dir (str): directory where to write tensorboard logs
        model_name (str): name for the model in tensorbboard
        rg_name (str): Name
        layer_used (str): _description_
        leaf_type (str): _description_
        k (int): _description_
        k_in (int): _description_
        batch_size (int, optional): _description_. Defaults to 100.
        lr (float, optional): _description_. Defaults to 0.01.
        max_num_epochs (int, optional): _description_. Defaults to 200.
        patience (int, optional): _description_. Defaults to 3.
        verbose (bool, optional): _description_. Defaults to True.
    """
    sqrt_eps = np.sqrt(torch.finfo(torch.get_default_dtype()).tiny)  # todo find better place
    pc_pf: TensorizedPC = integrate(pc)
    torch.set_default_tensor_type("torch.FloatTensor")

    # make experiment name string
    exp_name = "_".join([pc_hypar["DATA"], pc_hypar["RG"], pc_hypar["layer"], pc_hypar["LEAF"],
                         f"K_{pc_hypar['K']}", f"KIN_{pc_hypar['K_IN']}", f"lr_{pc_hypar['lr']}"])

    # model id uses date_time
    model_id: str = os.path.splitext(os.path.basename(save_path))[0]
    exp_name = f"{exp_name}_{model_id}"
    print("Experiment name: " + exp_name)

    # load data
    x_train, x_valid, x_test = load_dataset(dataset_name, device="cpu")

    optimizer = torch.optim.Adam([
        {'params': [p for p in pc.input_layer.parameters()]},
        {'params': [p for layer in pc.inner_layers for p in layer.parameters()], 'weight_decay': weight_decay}], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=1, eta_min=args.eta_min)

    # Setup Tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(save_path), model_id))

    # maybe creates save dir
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    def print_ll(tr_ll: float, val_ll: float, text: str):
        print(text, end="")
        print(f"\ttrain LL {tr_ll:.2f}", end="")
        print(f"\tvalid LL {val_ll:.2f}")


    """
    # Compute train and validation log-likelihood
     train_ll = eval_loglikelihood_batched(pc, x_train, device=device) / x_train.shape[0] if compute_train_ll else np.NAN
     valid_ll = eval_loglikelihood_batched(pc, x_valid, device=device) / x_valid.shape[0]

    print_ll(train_ll, valid_ll, "[Before Learning]")

    # ll on tensorboard
    if not np.isnan(train_ll):
        writer.add_scalar("train_ll", train_ll, 0)
    writer.add_scalar("valid_ll", valid_ll, 0)
    """

    # # # # # # # # # # # #
    # SETUP Early Stopping
    best_valid_ll = eval_loglikelihood_batched(pc, x_valid, device=device) / x_valid.shape[0]
    # best_test_ll = eval_loglikelihood_batched(pc, x_test, device=device) / x_test.shape[0]
    patience_counter = patience

    tik_train = time.time()
    for epoch_count in range(1, max_num_epochs + 1):

        # # # # # #
        #  Train  #
        # # # # # #
        idx_batches = torch.randperm(x_train.shape[0]).split(batch_size)

        # setup tqdm
        if verbose:
            pbar = tqdm(
                iterable=enumerate(idx_batches),
                total=len(idx_batches),
                unit="steps",
                ascii=" ▖▘▝▗▚▞█",
                ncols=120,
                )
        else:
            pbar = enumerate(idx_batches)

        train_ll = 0
        for batch_count, idx in pbar:
            batch_x = x_train[idx, :].unsqueeze(dim=-1).to(device)

            if batch_size == 1:
                batch_x = batch_x.reshape(1, -1)

            # compute ll
            log_likelihood = (pc(batch_x) - pc_pf(batch_x)).sum(dim=0)

            optimizer.zero_grad()
            (-log_likelihood).backward()

            # update with batch ll
            train_ll += log_likelihood.item()

            # CHECK
            check_validity_params(pc)
            # UPDATE
            optimizer.step()
            scheduler.step()

            # CHECK AGAIN
            check_validity_params(pc)

            # project params in inner layers TODO: remove or edit?
            if pc_hypar["REPARAM"] == "clamp":
                for layer in pc.inner_layers:
                    if type(layer) == CollapsedCPLayer:
                        layer.params_in().data = torch.clamp(layer.params_in(), min=sqrt_eps)
                    else:
                        layer.params().data = torch.clamp(layer.params(), min=sqrt_eps)

            if verbose:
                if batch_count % 10 == 0:
                    pbar.set_description(f"Epoch {epoch_count} Train LL={log_likelihood.item() / batch_size :.2f})")

        train_ll = train_ll / x_train.shape[0]
        valid_ll = eval_loglikelihood_batched(pc, x_valid, device=device) / x_valid.shape[0]

        print_ll(train_ll, valid_ll, f"[After epoch {epoch_count}]")
        if device != "cpu": print('Max allocated GPU: %.2f' % (torch.cuda.max_memory_allocated() / 1024 ** 3))

        # Not improved
        if valid_ll <= best_valid_ll:
            patience_counter -= 1
            if patience_counter == 0:
                print("-> Validation LL did not improve, early stopping")
                break

        else:
            # Improved, save model
            torch.save(pc, save_path)
            print("-> Saved model")

            # update best_valid_ll
            best_valid_ll = valid_ll
            # best_test_ll = eval_loglikelihood_batched(pc, x_test, device=device) / x_test.shape[0]
            patience_counter = patience

        writer.add_scalar("train_ll", train_ll, epoch_count)
        writer.add_scalar("valid_ll", valid_ll, epoch_count)
        writer.flush()

    print('Overall training time: %.2f (s)', time.time() - tik_train)
    # reload the model and compute test_ll
    pc = torch.load(save_path)
    best_train_ll = eval_loglikelihood_batched(pc, x_train, device=device) / x_train.shape[0]
    best_test_ll = eval_loglikelihood_batched(pc, x_test, device=device) / x_test.shape[0]

    print(args.reparam)
    print('train bpd: ', bpd_from_ll(pc, best_train_ll))
    print('valid bpd: ', bpd_from_ll(pc, best_valid_ll))
    print('test  bpd: ', bpd_from_ll(pc, best_test_ll))

    writer.add_hparams(
        hparam_dict=pc_hypar,
        metric_dict={
            "Best/Valid/ll": float(best_valid_ll),
            "Best/Valid/bpd": float(bpd_from_ll(pc, best_valid_ll)),
            "Best/Test/ll": float(best_test_ll),
            "Best/Test/bpd": float(bpd_from_ll(pc, best_test_ll)),
        },
        hparam_domain_discrete={
            "DATA": ["mnist", "fashion_mnist", "celeba"],
            "RG": ["QG", "PD", "QT"],
            "layer": ["cp", "cpshared", "tucker"],
            "LEAF": ["bin", "cat"],
            "REPARAM": ["softplus", "exp", "exp_temp", "relu"]
        },
    )
    writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("MNIST experiments.")
    parser.add_argument("--seed",           type=int,   default=42,         help="Random seed")
    parser.add_argument("--gpu",            type=int,   default=None,       help="Device on which run the benchmark")
    parser.add_argument("--dataset",        type=str,   default="mnist",    help="Dataset for the experiment")
    parser.add_argument("--model-dir",      type=str,   default="out",      help="Base dir for saving the model")
    parser.add_argument("--lr",             type=float, default=0.1,        help="Path of the model to be loaded")
    parser.add_argument("--patience",       type=int,   default=5,          help='patience for early stopping')
    parser.add_argument("--weight-decay",   type=float, default=0,          help="Weight decay coefficient")
    parser.add_argument("--num-sums",       type=int,   default=128,        help="Num sums")
    parser.add_argument("--num-input",      type=int,   default=None,       help="Num input distributions per leaf, if None then is equal to num-sums",)
    parser.add_argument("--rg",             type=str,   default="QT",       help="Region graph: 'PD', 'QG', or 'QT'")
    parser.add_argument("--layer",          type=str,                       help="Layer type: 'tucker', 'cp' or 'cp-shared'")
    parser.add_argument("--leaf",           type=str,                       help="Leaf type: either 'cat' or 'bin'")
    parser.add_argument("--reparam",        type=str,   default="exp",      help="Either 'exp', 'relu', or 'exp_temp'")
    parser.add_argument("--max-num-epochs", type=int,   default=200,        help="Max num epoch")
    parser.add_argument("--batch-size",     type=int,   default=128,        help="batch size")
    parser.add_argument("--train-ll",       type=bool,  default=False,      help="Compute train-ll at the end of each epoch")
    parser.add_argument("--progressbar",    type=bool,  default=False,      help="Print the progress bar")
    parser.add_argument('-t0',              type=int,   default=1,          help='sched CAWR t0, 1 for fixed lr ')
    parser.add_argument('-eta_min',         type=float, default=1e-4,       help='sched CAWR eta min')
    args = parser.parse_args()
    print(args)
    init_random_seeds(seed=args.seed)

    LAYER_TYPES = {
        "tucker": TuckerLayer,
        "cp": CollapsedCPLayer,
        "cp-shared": SharedCPLayer,
    }
    LEAF_TYPES = {"cat": CategoricalLayer, "bin": BinomialLayer}
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
    assert args.rg in ["QG", "QT", "PD"]
    assert args.leaf in LEAF_TYPES
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu"
    if args.num_input is None: args.num_input = args.num_sums

    rg: RegionGraph = {
        "QG": QuadTree(width=28, height=28, struct_decomp=False),
        "GT": QuadTree(width=28, height=28, struct_decomp=True),
        "PD": PoonDomingos(shape=(28, 28), delta=4)
    }[args.rg]

    # Setup leaves setting
    efamily_kwargs: dict
    if args.leaf == "cat":
        efamily_kwargs = {"num_categories": 256}
    elif args.leaf == "bin":
        efamily_kwargs = {"n": 256}

    REPARAM_TYPES["exp_temp"] = ReparamExpTemp
    REPARAM_TYPES["softmax_temp"] = ReparamSoftmaxTemp

    # Create probabilistic circuit
    pc = TensorizedPC.from_region_graph(
        rg=rg,
        layer_cls=LAYER_TYPES[args.layer],
        efamily_cls=LEAF_TYPES[args.leaf],
        efamily_kwargs=efamily_kwargs,
        num_inner_units=args.num_sums,
        num_input_units=args.num_input,
        reparam=REPARAM_TYPES[args.reparam],
    )
    pc.to(device)
    print(f"Num of params: {num_of_params(pc)}")
    model_name = args.layer

    # compose model path
    save_path = os.path.join(
        args.model_dir,
        args.dataset,
        args.rg,
        args.layer,
        args.leaf,
        args.reparam,
        str(args.num_sums),
        str(args.lr),
        get_date_time_str() + ".mdl",
    )

    train_procedure(
        pc=pc,
        pc_hypar={
            "RG": args.rg,
            "layer": args.layer,
            "LEAF": args.leaf,
            "REPARAM": args.reparam,
            "K": args.num_sums,
            "K_IN": args.num_input,
            "DATA": args.dataset,
            "lr": args.lr,
            "optimizer": "Adam",
            "batch_size": args.batch_size,
            "num_par": num_of_params(pc)
            },
        dataset_name=args.dataset,
        save_path=save_path,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_num_epochs=args.max_num_epochs,
        patience=args.patience,
        compute_train_ll=args.train_ll,
        verbose=args.progressbar
    )
