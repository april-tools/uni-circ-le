import argparse

import torch

from cirkit.layers.input.exp_family import CategoricalLayer
from cirkit.layers.sum_product import CPCollapsedLayer, CPSharedLayer, TuckerLayer
from cirkit.models.tensorized_circuit import TensorizedPC
from cirkit.region_graph import RegionGraph
from cirkit.region_graph.poon_domingos import PoonDomingos
from cirkit.region_graph.quad_tree import QuadTree
from measures import eval_bpd
from training import train_procedure
from utils import init_random_seeds, load_dataset, num_of_params

LAYER_TYPES = {
    "tucker": TuckerLayer,
    "cp": CPCollapsedLayer,
    "cp-shared": CPSharedLayer,
}
RG_TYPES = {"QG", "QT", "PD"}

if __name__ == "__main__":
    init_random_seeds()

    PARSER = argparse.ArgumentParser("MNIST experiments.")
    PARSER.add_argument(
        "--gpu", type=int, default=None, help="Device on which run the benchmark"
    )
    PARSER.add_argument(
        "--dataset", type=str, default="mnist", help="Dataset for the experiment"
    )
    PARSER.add_argument(
        "--model-dir", type=str, default=None, help="Dir for saving the model"
    )
    PARSER.add_argument(
        "--lr", type=float, default=0.1, help="Path of the model to be loaded"
    )
    PARSER.add_argument("--num-sums", type=int, default=128, help="Num sums")
    PARSER.add_argument(
        "--num-input", type=int, default=None, help="Num input distributions"
    )
    PARSER.add_argument("--rg", default="quad_tree", help="Region graph used")
    PARSER.add_argument("--layer", type=str, help="Layer type")
    PARSER.add_argument("--max-num-epochs", type=int, default=200, help="Max num epoch")
    PARSER.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for optimization"
    )
    PARSER.add_argument(
        "--tensorboard-dir", default=None, type=str, help="Path for tensorboard"
    )
    ARGS = PARSER.parse_args()

    DEVICE = (
        f"cuda:{ARGS.gpu}"
        if torch.cuda.is_available() and ARGS.gpu is not None
        else "cpu"
    )
    assert ARGS.layer in LAYER_TYPES
    assert ARGS.rg in RG_TYPES

    model_dir = ARGS.model_dir
    train_x, valid_x, test_x = load_dataset(ARGS.dataset, device=DEVICE)

    rg: RegionGraph
    if ARGS.rg == "QG":
        rg = QuadTree(width=28, height=28, struct_decomp=False)
    elif ARGS.rg == "QT":
        rg = QuadTree(width=28, height=28, struct_decomp=True)
    elif ARGS.rg == "PD":
        rg = PoonDomingos(shape=(28, 28), delta=4)
    else:
        raise AssertionError("Invalid RG")

    pc = TensorizedPC.from_region_graph(
        rg=rg,
        layer_cls=LAYER_TYPES[ARGS.layer],
        efamily_cls=CategoricalLayer,
        layer_kwargs={"prod_exp": True},
        efamily_kwargs={"num_categories": 256},
        num_inner_units=ARGS.num_sums,
        num_input_units=ARGS.num_input,
    )
    pc.to(DEVICE)
    print(f"Num of params: {num_of_params(pc)}")
    model_name = ARGS.layer

    train_procedure(
        pc=pc,
        dataset_name=ARGS.dataset,
        model_dir=model_dir,
        tensorboard_dir=ARGS.tensorboard_dir,
        model_name=model_name,
        rg_name=ARGS.rg,
        layer_used=ARGS.layer,
        k=ARGS.num_sums,
        k_in=ARGS.num_sums if ARGS.num_input is None else ARGS.num_input,
        prod_exp=True,
        batch_size=ARGS.batch_size,
        lr=ARGS.lr,
        max_num_epochs=ARGS.max_num_epochs,
        patience=10,
    )

    print(eval_bpd(pc, test_x))
