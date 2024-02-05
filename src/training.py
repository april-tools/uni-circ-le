import os
import sys
import argparse
from typing import Literal
import torch

sys.path.append(os.path.join(os.getcwd(), "cirkit"))
sys.path.append(os.path.join(os.getcwd(), "src"))

from torch.utils.tensorboard import SummaryWriter
from utils import *
from measures import *
from tqdm import tqdm

# cirkit
from cirkit.models.functional import integrate
from cirkit.models.tensorized_circuit import TensorizedPC
from cirkit.reparams.leaf import ReparamExp, ReparamLeaf
from cirkit.layers.input.exp_family.categorical import CategoricalLayer
from cirkit.layers.input.exp_family.binomial import BinomialLayer
from cirkit.layers.sum_product import CollapsedCPLayer, TuckerLayer, SharedCPLayer
from cirkit.models.tensorized_circuit import TensorizedPC
from cirkit.region_graph import RegionGraph
from cirkit.region_graph.poon_domingos import PoonDomingos
from cirkit.region_graph.quad_tree import QuadTree

class ReparamReLU(ReparamLeaf):
    def forward(self) -> torch.Tensor:
        return torch.relu(self.param)

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
    "softplus": ReparamSoftplus
    # "exp_temp" will be added at run-time
}





def train_procedure(
    *,
    pc: TensorizedPC,
    pc_hypar: dict,
    dataset_name: str,
    save_path: str,
    tensorboard_dir: str = "runs",
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
    writer = SummaryWriter(
        log_dir=os.path.join(os.getcwd(), f"{tensorboard_dir}/{exp_name}")
    )

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
            # for layer in pc.inner_layers:
            #    layer.clamp_params()

            if verbose:
                if batch_count % 10 == 0:
                    pbar.set_description(f"Epoch {epoch_count} Train LL={objective.item() / batch_size :.2f})")

        if not np.isnan(train_ll):
            train_ll = eval_loglikelihood_batched(pc, x_train) / x_train.shape[0]
        valid_ll = eval_loglikelihood_batched(pc, x_valid) / x_valid.shape[0]

        print_ll(train_ll, valid_ll, f"[After epoch {epoch_count}]")

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

    # reload the model and compute test_ll
    pc = torch.load(save_path)
    pc_pf = integrate(pc)
    best_test_ll = eval_loglikelihood_batched(pc, x_test) / x_test.shape[0]

    writer.add_hparams(
        hparam_dict=pc_hypar,
        metric_dict = {
            "Best/Valid/ll": float(best_valid_ll),
            "Best/Valid/bpd": float(bpd_from_ll(pc, best_valid_ll)),
            "Best/Test/ll": float(best_test_ll),
            "Best/Test/bpd": float(bpd_from_ll(pc, best_test_ll)),
        },
        hparam_domain_discrete = {
            "DATA": ["mnist", "fashion_mnist", "celeba"],
            "RG": ["QG", "PD", "QT"],
            "PAR": ["cp", "cpshared", "tucker"],
        },
    )
    writer.close()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser("MNIST experiments.")
    PARSER.add_argument("--seed", type=int, default=42, help="Random seed")
    PARSER.add_argument(
        "--gpu", type=int, default=None, help="Device on which run the benchmark"
    )
    PARSER.add_argument(
        "--dataset", type=str, default="mnist", help="Dataset for the experiment"
    )
    PARSER.add_argument(
        "--model-dir", type=str, default="out", help="Base dir for saving the model"
    )
    PARSER.add_argument(
        "--lr", type=float, default=0.1, help="Path of the model to be loaded"
    )
    PARSER.add_argument("--num-sums", type=int, default=128, help="Num sums")
    PARSER.add_argument(
        "--num-input",
        type=int,
        default=None,
        help="Num input distributions per leaf, if None then is equal to num-sums",
    )
    PARSER.add_argument(
        "--rg", default="quad_tree", help="Region graph: either 'PD', 'QG', or 'QT'"
    )
    PARSER.add_argument(
        "--layer", type=str, help="Layer type: either 'tucker', 'cp' or 'cp-shared'"
    )
    PARSER.add_argument("--leaf", type=str, help="Leaf type: either 'cat' or 'bin'")
    PARSER.add_argument("--reparam", type=str, default="exp", help="Either 'exp', 'relu', or 'exp_temp'")
    PARSER.add_argument("--max-num-epochs", type=int, default=200, help="Max num epoch")
    PARSER.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for optimization"
    )
    PARSER.add_argument(
        "--tensorboard-dir", default="runs", type=str, help="Path for tensorboard"
    )
    PARSER.add_argument("--train-ll", type=bool, default=False, help="Compute train-ll at the end of each epoch")
    PARSER.add_argument("--progressbar", type=bool, default=False, help="Print the progress bar")
    ARGS = PARSER.parse_args()

    init_random_seeds(seed=ARGS.seed)

    DEVICE = (
        f"cuda:{ARGS.gpu}"
        if torch.cuda.is_available() and ARGS.gpu is not None
        else "cpu"
    )
    assert ARGS.layer in LAYER_TYPES
    assert ARGS.rg in RG_TYPES
    assert ARGS.leaf in LEAF_TYPES
    if ARGS.num_input is None:
        ARGS.num_input = ARGS.num_sums

    train_x, valid_x, test_x = load_dataset(ARGS.dataset, device=DEVICE)

    # Setup region graph
    rg: RegionGraph
    if ARGS.rg == "QG":  # TODO: generalize width and height
        rg = QuadTree(width=28, height=28, struct_decomp=False)
    elif ARGS.rg == "QT":
        rg = QuadTree(width=28, height=28, struct_decomp=True)
    elif ARGS.rg == "PD":
        rg = PoonDomingos(shape=(28, 28), delta=4)
    else:
        raise AssertionError("Invalid RG")

    # Setup leaves setting
    efamily_kwargs: dict
    if ARGS.leaf == "cat":
        efamily_kwargs = {"num_categories": 256}
    elif ARGS.leaf == "bin":
        efamily_kwargs = {"n": 256}

    # setup reparam
    class ReparamExpTemp(ReparamLeaf):
        def forward(self) -> torch.Tensor:
            return torch.exp(self.param / np.sqrt(ARGS.num_sums))
    REPARAM_TYPES["exp_temp"] = ReparamExpTemp

    # Create probabilistic circuit
    pc = TensorizedPC.from_region_graph(
        rg=rg,
        layer_cls=LAYER_TYPES[ARGS.layer],
        efamily_cls=LEAF_TYPES[ARGS.leaf],
        efamily_kwargs=efamily_kwargs,
        num_inner_units=ARGS.num_sums,
        num_input_units=ARGS.num_input,
        reparam=REPARAM_TYPES[ARGS.reparam],
    )
    pc.to(DEVICE)
    print(f"Num of params: {num_of_params(pc)}")
    model_name = ARGS.layer

    # compose model path
    # e.g. out/mnist/
    save_path = os.path.join(
        ARGS.model_dir,
        ARGS.dataset,
        ARGS.rg,
        ARGS.layer,
        ARGS.leaf,
        ARGS.reparam,
        str(ARGS.num_sums),
        str(ARGS.lr),
        get_date_time_str() + ".mdl",
    )

    # Train the model
    train_procedure(
        pc=pc,
        pc_hypar={
            "RG": ARGS.rg,
            "PAR": ARGS.layer,
            "LEAF": ARGS.leaf,
            "K": ARGS.num_sums,
            "K_IN": ARGS.num_input,
            "DATA": ARGS.dataset,
            "lr": ARGS.lr,
            "optimizer": "Adam",
            "batch_size": ARGS.batch_size,
            "num_par": num_of_params(pc)
            },
        dataset_name=ARGS.dataset,
        save_path=save_path,
        tensorboard_dir=ARGS.tensorboard_dir,
        batch_size=ARGS.batch_size,
        lr=ARGS.lr,
        max_num_epochs=ARGS.max_num_epochs,
        patience=10,
        compute_train_ll=ARGS.train_ll,
        verbose=ARGS.progressbar
    )

    print(eval_bpd(pc, test_x))
