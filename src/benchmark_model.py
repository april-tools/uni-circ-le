import argparse
import enum
import functools
from dataclasses import dataclass
from typing import List, Literal, Tuple
import sys
import os


sys.path.append(os.path.join(os.getcwd(), "cirkit"))
sys.path.append(os.path.join(os.getcwd(), "src"))

import numpy as np
import pandas as pd
import torch
from torch import Tensor, optim
from torch.utils.data import DataLoader, TensorDataset

from benchmark.utils.gpu_benchmark import benchmarker
from cirkit.layers.input.exp_family import CategoricalLayer
from cirkit.layers.sum_product.tucker import TuckerLayer
from cirkit.layers.sum_product.cp import CollapsedCPLayer, SharedCPLayer
from cirkit.models import TensorizedPC
from cirkit.region_graph import RegionGraph
from cirkit.utils import RandomCtx, set_determinism
from cirkit.region_graph import RegionGraph
from cirkit.region_graph.poon_domingos import PoonDomingos
from cirkit.region_graph.quad_tree import QuadTree
from real_qt import RealQuadTree
from clt import tree2rg
from trees import TREE_DICT




# device: torch.device("cuda")


class _Modes(str, enum.Enum):  # TODO: StrEnum introduced in 3.11
    """Execution modes."""

    TRAIN = "train"
    EVAL = "eval"


@dataclass
class _ArgsNamespace(argparse.Namespace):
    mode: _Modes = _Modes.TRAIN
    seed: int = 42
    num_batches: int = 20
    batch_size: int = 128
    region_graph: str = ""
    layer: str = ""
    num_latents: int = 32  # TODO: rename this
    first_pass_only: bool = False
    gpu: int = 0
    results_csv: str = ""


def process_args() -> _ArgsNamespace:
    """Process command line arguments.

    Returns:
        ArgsNamespace: Parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=_Modes, choices=_Modes.__members__.values(), help="mode")
    parser.add_argument("--seed", type=int, help="seed, 0 for disable")
    parser.add_argument("--num_batches", type=int, help="num_batches")
    parser.add_argument("--batch_size", type=int, help="batch_size")
    parser.add_argument("--region_graph", type=str, help="region_graph to use")
    parser.add_argument("--layer", type=str, help="Either 'cp', 'cp-shared' or 'tucker'")
    parser.add_argument("--num_latents", type=int, help="num_latents")
    parser.add_argument("--first_pass_only", action="store_true", help="first_pass_only")
    parser.add_argument("--gpu", type=int, help="Which gpu to use")
    parser.add_argument("--results-csv", type=str, help="Path of the results csv (will be created if inexistent)")
    return parser.parse_args(namespace=_ArgsNamespace())




@torch.no_grad()
def evaluate(
    pc: TensorizedPC, data_loader: DataLoader[Tuple[Tensor, ...]], device
) -> Tuple[Tuple[List[float], List[float]], float]:
    """Evaluate circuit on given data.

    Args:
        pc (TensorizedPC): The PC to evaluate.
        data_loader (DataLoader[Tuple[Tensor, ...]]): The evaluation data.

    Returns:
        Tuple[Tuple[List[float], List[float]], float]:
         A tuple consisting of time and memory measurements, and the average LL.
    """

    def _iter(x: Tensor) -> Tensor:
        return pc(x)

    ll_total = 0.0
    ts, ms = [], []
    batch: Tuple[Tensor]
    for batch in data_loader:
        x = batch[0].to(device)
        ll, (t, m) = benchmarker(functools.partial(_iter, x), device=device)
        ts.append(t)
        ms.append(m)
        ll_total += ll.mean().item()
        del x, ll
    return (ts, ms), ll_total / len(data_loader)


def train(
    pc: TensorizedPC, optimizer: optim.Optimizer, data_loader: DataLoader[Tuple[Tensor, ...]], device
) -> Tuple[Tuple[List[float], List[float]], float]:
    """Train circuit on given data.

    Args:
        pc (TensorizedPC): The PC to optimize.
        optimizer (optim.Optimizer): The optimizer for circuit.
        data_loader (DataLoader[Tuple[Tensor, ...]]): The training data.

    Returns:
        Tuple[Tuple[List[float], List[float]], float]:
         A tuple consisting of time and memory measurements, and the average LL.
    """

    def _iter(x: Tensor) -> Tensor:
        optimizer.zero_grad()
        ll = pc(x)
        ll = ll.mean()
        (-ll).backward()  # type: ignore[no-untyped-call]  # we optimize NLL
        optimizer.step()
        return ll.detach()

    ll_total = 0.0
    ts, ms = [], []
    batch: Tuple[Tensor]
    for batch in data_loader:
        x = batch[0].to(device)
        ll, (t, m) = benchmarker(functools.partial(_iter, x), device=device)
        ts.append(t)
        ms.append(m)
        ll_total += ll.item()
        del x, ll  # TODO: is everything released properly
    return (ts, ms), ll_total / len(data_loader)


def do_benchmarking(args, mode: _Modes) -> None:
    """Execute the main procedure."""

    assert args.region_graph, "Must provide a RG filename"
    if args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(args)

    if args.seed:
        # TODO: find a way to set w/o with
        RandomCtx(args.seed).__enter__()  # pylint: disable=unnecessary-dunder-call
        set_determinism(check_hash_seed=False)

    num_vars = 28 * 28
    data_size = args.batch_size if args.first_pass_only else args.num_batches * args.batch_size
    rand_data = torch.randint(256, (data_size, num_vars, 1), dtype=torch.float32)
    data_loader = DataLoader(
        dataset=TensorDataset(rand_data),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # create RG
    REGION_GRAPHS = {
    'QG':   QuadTree(width=28, height=28, struct_decomp=False),
    'QT':   QuadTree(width=28, height=28, struct_decomp=True),
    'PD':   PoonDomingos(shape=(28, 28), delta=4),
    'CLT':  tree2rg(TREE_DICT["mnist"]),
    'RQT':  RealQuadTree(width=28, height=28)
    }

    # choose layer
    LAYER_TYPES = {
        "tucker": TuckerLayer,
        "cp": CollapsedCPLayer,
        "cp-shared": SharedCPLayer,
    }

    assert args.region_graph in REGION_GRAPHS
    rg: RegionGraph = REGION_GRAPHS[args.region_graph]
    layer = LAYER_TYPES[args.layer]

    pc = TensorizedPC.from_region_graph(
        rg,
        layer_cls=layer,
        efamily_cls=CategoricalLayer,
        efamily_kwargs={"num_categories": 256},  # type: ignore[misc]
        num_inner_units=args.num_latents,
        num_input_units=args.num_latents,
    )
    pc.to(device)
    print(pc)
    print(f"Number of parameters: {sum(p.numel() for p in pc.parameters())}")

    if mode == _Modes.TRAIN:
        optimizer = optim.Adam(pc.parameters())  # just keep everything default
        (ts, ms), ll_train = train(pc, optimizer, data_loader, device=device)
        print("Train LL:", ll_train)
    elif mode == _Modes.EVAL:
        (ts, ms), ll_eval = evaluate(pc, data_loader, device=device)
        print("Evaluation LL:", ll_eval)
    else:
        assert False, "Something is wrong here"
    if not args.first_pass_only and args.num_batches > 1:
        # Skip warmup step
        ts, ms = ts[1:], ms[1:]
    mu_t, sigma_t = np.mean(ts).item(), np.std(ts).item()  # type: ignore[misc]
    mu_m, sigma_m = np.mean(ms).item(), np.std(ms).item()  # type: ignore[misc]

    print(f"Time (ms): {mu_t:.3f}+-{sigma_t:.3f}")
    print(f"Memory (MiB): {mu_m:.3f}+-{sigma_m:.3f}")

    return mu_t, sigma_t, mu_m, sigma_m


if __name__ == "__main__":
    args = process_args()

    # benchmark evaluation mode
    eval_mu_t, eval_sigma_t, eval_mu_m, eval_sigma_m = do_benchmarking(args, mode="eval")
    # benchmark training mode
    train_mu_t, train_sigma_t, train_mu_m, train_sigma_m = do_benchmarking(args, mode="train")

    csv_row = {
        "rg": args.region_graph,
        "layer": args.layer,
        "k": args.num_latents,
        "batch_size": args.batch_size,
        "eval_time": eval_mu_t,
        "eval_time_std": eval_sigma_t,
        "eval_space": eval_mu_m,
        "eval_space_std": eval_sigma_m,
        "train_time": train_mu_t,
        "train_time_std": train_sigma_t,
        "train_space": train_mu_m,
        "train_space_std": train_mu_t
    }

    df = pd.DataFrame.from_dict([csv_row])

    if os.path.exists(args.results_csv):
        df.to_csv(args.results_csv, mode="a", index=False)
    else:
        df.to_csv(args.results_csv, index=False)
