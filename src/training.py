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
from cirkit.reparams.leaf import ReparamExp, ReparamIdentity, ReparamLeaf
from cirkit.layers.input.exp_family.categorical import CategoricalLayer
from cirkit.layers.input.exp_family.binomial import BinomialLayer
from cirkit.layers.sum_product import CollapsedCPLayer, TuckerLayer, SharedCPLayer
from cirkit.models.tensorized_circuit import TensorizedPC
from cirkit.region_graph import RegionGraph
from cirkit.region_graph.poon_domingos import PoonDomingos
from cirkit.region_graph.quad_tree import QuadTree

class ReparamReLU(ReparamLeaf):
    eps = torch.finfo(torch.get_default_dtype()).tiny
    def forward(self) -> torch.Tensor:
        return torch.clamp(self.param, min=ReparamReLU.eps)

class ReparamSoftplus(ReparamLeaf):
    def forward(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.param)

LAYER_TYPES = {
    "tucker": TuckerLayer,
    "cp": CollapsedCPLayer,
    "cp-shared": SharedCPLayer,
}
RG_TYPES = {"QG", "QT", "PD"}
LEAF_TYPES = {"cat": CategoricalLayer, "bin": BinomialLayer}
REPARAM_TYPES = {
    "exp": ReparamExp,
    "relu": ReparamReLU,
    "softplus": ReparamSoftplus,
    "clamp": ReparamIdentity
    # "exp_temp" will be added at run-time
}


def train_procedure(
    *,
    pc: TensorizedPC,
    pc_hypar: dict,
    dataset_name: str,
    save_path: str,
    batch_size=128,
    lr=0.01,
    max_num_epochs=200,
    patience=10,
    verbose=True,
    compute_train_ll=False
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
    pc_pf = integrate(pc)
    torch.set_default_tensor_type("torch.FloatTensor")

    # make experiment name string
    exp_name = "_".join([pc_hypar["DATA"], pc_hypar["RG"], pc_hypar["PAR"], pc_hypar["LEAF"],
                         f"K_{pc_hypar['K']}", f"KIN_{pc_hypar['K_IN']}", f"lr_{pc_hypar['lr']}"])

    exp_name = f"{exp_name}_{get_date_time_str()}"
    print("Experiment name: " + exp_name)

    # load data
    x_train, x_valid, x_test = load_dataset(dataset_name, device=get_pc_device(pc))
    # load optimizer
    optimizer = torch.optim.Adam(pc.parameters(), lr=lr)

    # Setup Tensorboard writer
    writer = SummaryWriter(log_dir=os.path.dirname(save_path))

    # maybe creates save dir
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # Compute train and validation log-likelihood
    train_ll = eval_loglikelihood_batched(pc, x_train) / x_train.shape[0] if compute_train_ll else np.NAN
    valid_ll = eval_loglikelihood_batched(pc, x_valid) / x_valid.shape[0]

    def print_ll(tr_ll: float, val_ll: float, text: str):
        print(text, end="")
        if not np.isnan(tr_ll):
            print(f"\ttrain LL {tr_ll}", end="")
        print(f"\tvalid LL {val_ll}")

    print_ll(train_ll, valid_ll, "[Before Learning]")

    # ll on tensorboard
    if not np.isnan(train_ll):
        writer.add_scalar("train_ll", train_ll, 0)
    writer.add_scalar("valid_ll", valid_ll, 0)

    # # # # # # # # # # # #
    # SETUP Early Stopping
    best_valid_ll = valid_ll
    # best_test_ll = eval_loglikelihood_batched(pc, x_test) / x_test.shape[0]
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

        for batch_count, idx in pbar:
            batch_x = x_train[idx, :].unsqueeze(dim=-1)

            if batch_size == 1:
                batch_x = batch_x.reshape(1, -1)

            log_likelihood = (pc(batch_x) - pc_pf(batch_x)).sum(0)
            objective = -log_likelihood
            optimizer.zero_grad()
            objective.backward()

            # CHECK
            check_validity_params(pc)
            # UPDATE
            optimizer.step()
            # CHECK AGAIN
            check_validity_params(pc)

            # project params in inner layers TODO: remove or edit?
            if pc_hypar["REPARAM"] == "clamp":
                eps = torch.finfo(torch.get_default_dtype()).tiny
                for layer in pc.inner_layers:
                    if type(layer) == CollapsedCPLayer:
                        layer.params_in().data = torch.clamp(layer.params_in(), min=ReparamReLU.eps)
                    else:
                        layer.params().data = torch.clamp(layer.params(), min=ReparamReLU.eps)

            if verbose:
                if batch_count % 10 == 0:
                    pbar.set_description(f"Epoch {epoch_count} Train LL={objective.item() / batch_size :.2f})")

        if not np.isnan(train_ll):
            train_ll = eval_loglikelihood_batched(pc, x_train) / x_train.shape[0]
        valid_ll = eval_loglikelihood_batched(pc, x_valid) / x_valid.shape[0]

        print_ll(train_ll, valid_ll, f"[After epoch {epoch_count}]")
        if device != "cpu": print('Max allocated GPU: %.2f', torch.cuda.max_memory_allocated() / 1024 ** 3)

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
            # best_test_ll = eval_loglikelihood_batched(pc, x_test) / x_test.shape[0]
            patience_counter = patience

        if not np.isnan(train_ll):
            writer.add_scalar("train_ll", train_ll, epoch_count)
        writer.add_scalar("valid_ll", valid_ll, epoch_count)
        writer.flush()

    print('Overall training time: %.2f (s)', time.time() - tik_train)
    # reload the model and compute test_ll
    pc = torch.load(save_path)
    best_test_ll = eval_loglikelihood_batched(pc, x_test) / x_test.shape[0]

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
            "PAR": ["cp", "cpshared", "tucker"],
            "LEAF": ["bin", "cat"],
            "REPARAM": ["softplus", "exp", "exp_temp", "relu"]
        },
    )
    writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("MNIST experiments.")
    parser.add_argument("--seed",           type=int,   default=42,             help="Random seed")
    parser.add_argument("--gpu",            type=int,   default=None,           help="Device on which run the benchmark")
    parser.add_argument("--dataset",        type=str,   default="mnist",        help="Dataset for the experiment")
    parser.add_argument("--model-dir",      type=str,   default="out",          help="Base dir for saving the model")
    parser.add_argument("--lr",             type=float, default=0.1,            help="Path of the model to be loaded")
    parser.add_argument("--num-sums",       type=int,   default=128,            help="Num sums")
    parser.add_argument("--num-input",      type=int,   default=None,           help="Num input distributions per leaf, if None then is equal to num-sums",)
    parser.add_argument( "--rg",            type=str,   default="quad_tree",    help="Region graph: 'PD', 'QG', or 'QT'")
    parser.add_argument("--layer",          type=str,                           help="Layer type: 'tucker', 'cp' or 'cp-shared'")
    parser.add_argument("--leaf",           type=str,                           help="Leaf type: either 'cat' or 'bin'")
    parser.add_argument("--reparam",        type=str,   default="exp",          help="Either 'exp', 'relu', or 'exp_temp'")
    parser.add_argument("--max-num-epochs", type=int,   default=200,            help="Max num epoch")
    parser.add_argument("--batch-size",     type=int,   default=128,            help="batch size")
    parser.add_argument("--train-ll",       type=bool,  default=False,          help="Compute train-ll at the end of each epoch")
    parser.add_argument("--progressbar",    type=bool,  default=False,          help="Print the progress bar")
    args = parser.parse_args()
    print(args)
    init_random_seeds(seed=args.seed)

    device = (
        f"cuda:{args.gpu}"
        if torch.cuda.is_available() and args.gpu is not None
        else "cpu"
    )
    assert args.layer in LAYER_TYPES
    assert args.rg in RG_TYPES
    assert args.leaf in LEAF_TYPES
    if args.num_input is None:
        args.num_input = args.num_sums

    train_x, valid_x, test_x = load_dataset(args.dataset, device="cpu")

    # Setup region graph
    rg: RegionGraph
    if args.rg == "QG":  # TODO: generalize width and height
        rg = QuadTree(width=28, height=28, struct_decomp=False)
    elif args.rg == "QT":
        rg = QuadTree(width=28, height=28, struct_decomp=True)
    elif args.rg == "PD":
        rg = PoonDomingos(shape=(28, 28), delta=4)
    else:
        raise AssertionError("Invalid RG")

    # Setup leaves setting
    efamily_kwargs: dict
    if args.leaf == "cat":
        efamily_kwargs = {"num_categories": 256}
    elif args.leaf == "bin":
        efamily_kwargs = {"n": 256}

    # setup reparam
    class ReparamExpTemp(ReparamLeaf):
        def forward(self) -> torch.Tensor:
            return torch.exp(self.param / np.sqrt(args.num_sums))
    REPARAM_TYPES["exp_temp"] = ReparamExpTemp

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
    # e.g. out/mnist/
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

    # Train the model
    train_procedure(
        pc=pc,
        pc_hypar={
            "RG": args.rg,
            "PAR": args.layer,
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
        max_num_epochs=args.max_num_epochs,
        patience=10,
        compute_train_ll=args.train_ll,
        verbose=args.progressbar
    )

    print(args.reparam)
    print('train bpd: ', eval_bpd(pc, train_x))
    print('valid bpd: ', eval_bpd(pc, valid_x))
    print('test  bpd: ', eval_bpd(pc, test_x))
