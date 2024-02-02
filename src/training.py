import os
import sys
import argparse
import torch

sys.path.append("src")
sys.path.append("cirkit")

from torch.utils.tensorboard import SummaryWriter
from utils import *
from measures import *
from tqdm import tqdm

# cirkit
from cirkit.models.functional import integrate
from cirkit.models.tensorized_circuit import TensorizedPC
from cirkit.reparams.leaf import ReparamExp
from cirkit.layers.input.exp_family.categorical import CategoricalLayer
from cirkit.layers.input.exp_family.binomial import BinomialLayer
from cirkit.layers.sum_product import CollapsedCPLayer, TuckerLayer, SharedCPLayer
from cirkit.models.tensorized_circuit import TensorizedPC
from cirkit.region_graph import RegionGraph
from cirkit.region_graph.poon_domingos import PoonDomingos
from cirkit.region_graph.quad_tree import QuadTree

LAYER_TYPES = {
    "tucker": TuckerLayer,
    "cp": CollapsedCPLayer,
    "cp-shared": SharedCPLayer,
}
RG_TYPES = {"QG", "QT", "PD"}
LEAF_TYPES = {"cat": CategoricalLayer, "bin": BinomialLayer}


def train_procedure(
    *,
    pc: TensorizedPC,
    dataset_name: str,
    save_path: str,
    tensorboard_dir: str,
    model_name: str,
    rg_name: str,
    layer_used: str,
    leaf_type: str,
    k: int,
    k_in: int,
    batch_size=100,
    lr=0.01,
    max_num_epochs=200,
    patience=3,
    verbose=True,
):
    """Train a TensorizedPC using gradient descent.

    Args:
        pc (TensorizedPC): _description_
        dataset_name (str): _description_
        save_path (str): _description_
        tensorboard_dir (str): _description_
        model_name (str): _description_
        rg_name (str): _description_
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
    hyperparams = {
        "rg": rg_name,
        "layer": layer_used,
        "dataset": dataset_name,
        "leaf": leaf_type,
        "lr": lr,
        "optimizer": "Adam",
        "batch_size": batch_size,
        "num_parameters": num_of_params(pc),
        "k": k,
        "k_in": k_in,
        "patience": patience,
    }

    hyp_in_name = {
        k: hyperparams[k] for k in {"dataset", "rg", "layer", "k", "k_in", "lr"}
    }
    experiment_name = "".join([f"{key}_{value}_" for key, value in hyp_in_name.items()])
    experiment_name += "".join(
        [
            f"{key}_{value:.3f}_"
            for key, value in hyp_in_name.items()
            if type(value) == float
        ]
    )
    experiment_name = f"{model_name}_" + experiment_name + f"_{get_date_time_str()}"
    assert experiment_name != ""
    print("Experiment name: " + experiment_name)

    x_train, x_valid, x_test = load_dataset(dataset_name, device=get_pc_device(pc))

    optimizer = torch.optim.Adam(pc.parameters(), lr=lr)  # check if is correct

    if tensorboard_dir is None:
        tensorboard_dir = "runs"
    writer = SummaryWriter(
        log_dir=os.path.join(os.getcwd(), f"{tensorboard_dir}/{experiment_name}")
    )
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    train_ll = eval_loglikelihood_batched(pc, x_train) / x_train.shape[0]
    valid_ll = eval_loglikelihood_batched(pc, x_valid) / x_valid.shape[0]

    if verbose:
        print("[Before Learning]")
        print("\ttrain LL {}   valid LL {}".format(train_ll, valid_ll))

    # ll on tensorboard
    writer.add_scalar("train_ll", train_ll, 0)
    writer.add_scalar("valid_ll", valid_ll, 0)

    ####
    # SETUP
    best_valid_ll = valid_ll
    best_test_ll = eval_loglikelihood_batched(pc, x_test) / x_test.shape[0]
    patience_counter = patience


    for epoch_count in range(1, max_num_epochs + 1):

        # # # # # #
        #   LEARN
        # # # # # #
        idx_batches = torch.randperm(x_train.shape[0]).split(batch_size)

        # setup tqdm
        pbar = tqdm(
            iterable=enumerate(idx_batches),
            total=len(idx_batches),
            unit="steps",
            ascii=" ▖▘▝▗▚▞█",
            ncols=120,
            )

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

            if batch_count % 10 == 0:
                    pbar.set_description(f"Epoch {epoch_count} Train LL={objective.item():.2f})")


        train_ll = eval_loglikelihood_batched(pc, x_train) / x_train.shape[0]
        valid_ll = eval_loglikelihood_batched(pc, x_valid) / x_valid.shape[0]

        if verbose:
            print(f"[After epoch {epoch_count}]")
            print("\ttrain LL {}   valid LL {}".format(train_ll, valid_ll))

        # Not improved
        if valid_ll <= best_valid_ll:
            patience_counter -= 1
            if patience_counter == 0:
                if verbose:
                    print("-> Validation LL did not improve, early stopping")
                break

        else:
            # Improved, save model
            torch.save(pc, save_path)
            if verbose:
                print("-> Saved model")

            # update best_valid_ll
            best_valid_ll = valid_ll
            best_test_ll = eval_loglikelihood_batched(pc, x_test) / x_test.shape[0]
            patience_counter = patience

        writer.add_scalar("train_ll", train_ll, epoch_count)
        writer.add_scalar("valid_ll", valid_ll, epoch_count)
        writer.flush()

    writer.add_hparams(
        hparam_dict=hyperparams,
        metric_dict={
            "Best/Valid/ll": best_valid_ll,
            "Best/Valid/bpd": bpd_from_ll(pc, best_valid_ll),
            "Best/Test/ll": float(best_test_ll),
            "Best/Test/bpd": float(bpd_from_ll(pc, best_test_ll)),
        },
        hparam_domain_discrete={
            "dataset": ["mnist", "fashion_mnist"],
            "rg": ["QG", "PD", "QT"],
            "layer": ["cp", "cp-shared", "tucker"],
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
    PARSER.add_argument("--max-num-epochs", type=int, default=200, help="Max num epoch")
    PARSER.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for optimization"
    )
    PARSER.add_argument(
        "--tensorboard-dir", default=None, type=str, help="Path for tensorboard"
    )
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

    # Create probabilistic circuit
    pc = TensorizedPC.from_region_graph(
        rg=rg,
        layer_cls=LAYER_TYPES[ARGS.layer],
        efamily_cls=LEAF_TYPES[ARGS.leaf],
        efamily_kwargs=efamily_kwargs,
        num_inner_units=ARGS.num_sums,
        num_input_units=ARGS.num_input,
        reparam=ReparamExp,
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
        str(ARGS.num_sums),
        str(ARGS.lr),
        get_date_time_str() + ".mdl",
    )

    # Train the model
    train_procedure(
        pc=pc,
        dataset_name=ARGS.dataset,
        save_path=save_path,
        tensorboard_dir=ARGS.tensorboard_dir,
        model_name=model_name,
        rg_name=ARGS.rg,
        layer_used=ARGS.layer,
        leaf_type=ARGS.leaf,
        k=ARGS.num_sums,
        k_in=ARGS.num_input,
        batch_size=ARGS.batch_size,
        lr=ARGS.lr,
        max_num_epochs=ARGS.max_num_epochs,
        patience=10,
    )

    print(eval_bpd(pc, test_x))
