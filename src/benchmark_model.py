import sys
import os
# sys.path.append(os.path.join(os.getcwd(), "src"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ten-pcs/')))

import argparse
import functools
from typing import List

from utils import count_pc_params, count_trainable_parameters

import numpy as np
import pandas as pd
from torch import Tensor, optim
from torch.utils.data import DataLoader, TensorDataset

from tenpcs.layers.input.exp_family import CategoricalLayer
from tenpcs.layers.sum_product.tucker import TuckerLayer
from tenpcs.layers.sum_product.cp import CollapsedCPLayer, SharedCPLayer
from tenpcs.region_graph.poon_domingos import PoonDomingos
from tenpcs.region_tree.quad_tree import QuadTree
from tenpcs.region_graph.quad_graph import QuadGraph
from tenpcs.models.tensorized_circuit import TensorizedPC
from tenpcs.layers.sum_product.cp_shared import ScaledSharedCPLayer


from typing import Callable, Tuple, TypeVar

import torch

T = TypeVar("T")


def timer(fn: Callable[[], T]) -> Tuple[T, float]:
    """Time a given function for GPU time cost.

    Args:
        fn (Callable[[], T]): The function to time.

    Returns:
        Tuple[T, float]: The original return value, and time in ms.
    """
    start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    start_event.record(torch.cuda.current_stream())  # type: ignore[no-untyped-call]
    ret = fn()
    end_event.record(torch.cuda.current_stream())  # type: ignore[no-untyped-call]
    torch.cuda.synchronize()  # wait for event to be recorded
    elapsed_time: float = start_event.elapsed_time(end_event)  # type: ignore[no-untyped-call]
    return ret, elapsed_time


def benchmarker(fn: Callable[[], T], device: torch.device) -> Tuple[T, Tuple[float, float]]:
    """Benchmark a given function for GPU time and (peak) memory cost.

    Args:
        fn (Callable[[], T]): The function to benchmark.

    Returns:
        Tuple[T, Tuple[float, float]]: The original return value, and time in ms and memory in MiB.
    """
    torch.cuda.synchronize()  # finish all previous work before resetting mem stats
    torch.cuda.reset_peak_memory_stats()
    ret, elapsed_time = timer(fn)
    peak_memory = torch.cuda.max_memory_allocated(device=device) / 1024 ** 3
    return ret, (elapsed_time, peak_memory)



@torch.no_grad()
def evaluate(
    pc: TensorizedPC,
    data_loader: DataLoader[Tuple[Tensor, ...]],
    device
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
    pc: TensorizedPC,
    data_loader: DataLoader[Tuple[Tensor, ...]],
    device
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
    optimizer = optim.Adam(pc.parameters())  # just keep everything default
    for batch in data_loader:
        x = batch[0].to(device)
        ll, (t, m) = benchmarker(functools.partial(_iter, x), device=device)
        ts.append(t)
        ms.append(m)
        ll_total += ll.item()
        del x, ll  # TODO: is everything released properly
    return (ts, ms), ll_total / len(data_loader)


def do_benchmarking(pc, data_loader, device, mode: str) -> None:
    """Execute the main procedure."""

    if mode == "train":
        (ts, ms), ll_train = train(pc, data_loader, device=device)
        print("Train LL:", ll_train)
    elif mode == "test":
        (ts, ms), ll_eval = evaluate(pc, data_loader, device=device)
        print("Evaluation LL:", ll_eval)
    else:
        assert False, "Something is wrong here"

    ts, ms = ts[1:], ms[1:]  # Skip warmup step
    mu_t, sigma_t = np.mean(ts).item(), np.std(ts).item()  # type: ignore[misc]
    mu_m, sigma_m = np.mean(ms).item(), np.std(ms).item()  # type: ignore[misc]

    print(f"{mode} time (ms): {mu_t:.3f}+-{sigma_t:.3f}")
    print(f"{mode} memory (GiB): {mu_m:.3f}+-{sigma_m:.3f}")

    return mu_t, sigma_t, mu_m, sigma_m


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Unused, required for grid experiment")
    parser.add_argument("--num-steps", type=int, default=250, help="num_steps")
    parser.add_argument("--batch-size", type=int, help="batch_size")
    parser.add_argument("--region-graph", type=str, help="region_graph to use")
    parser.add_argument("--layer", type=str, help="Either 'cp', 'cp-shared' or 'tucker'")
    parser.add_argument("--k", type=int, default=128, help="Num categories for mixtures")
    parser.add_argument("--gpu", type=int, help="Which gpu to use")
    parser.add_argument("--results-csv", type=str, help="Path of the results csv (will be created if inexistent)")
    args = parser.parse_args()
    print(args)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu"

    ######################################################################################
    ################################## create fake fata ##################################
    ######################################################################################

    if args.dataset == "celeba":
        SQUARE_SIZE = 64
        NUM_CHANNELS = 3
    else:
        SQUARE_SIZE = 28
        NUM_CHANNELS = 1

    num_vars = SQUARE_SIZE**2

    data_size = args.num_steps * args.batch_size
    rand_data = torch.randint(256, (data_size, num_vars, NUM_CHANNELS), dtype=torch.float32)
    data_loader = DataLoader(
        dataset=TensorDataset(rand_data),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    #######################################################################################
    ################################## instantiate model ##################################
    #######################################################################################

    # create RG
    REGION_GRAPHS = {
        'QG': lambda: QuadGraph(width=SQUARE_SIZE, height=SQUARE_SIZE),
        'PD': lambda: PoonDomingos(shape=(SQUARE_SIZE, SQUARE_SIZE), delta=4),
        'QT': lambda: QuadTree(width=SQUARE_SIZE, height=SQUARE_SIZE)
    }

    # choose layer
    LAYER_TYPES = {
        "tucker": TuckerLayer,
        "cp": CollapsedCPLayer,
        "cp-xs": SharedCPLayer,
        "cp-s": ScaledSharedCPLayer,
    }

    pc = TensorizedPC.from_region_graph(
        rg=REGION_GRAPHS[args.region_graph](),
        layer_cls=LAYER_TYPES[args.layer],
        efamily_cls=CategoricalLayer,
        efamily_kwargs={"num_categories": 256},  # type: ignore[misc]
        num_inner_units=args.k,
        num_input_units=args.k,
        num_channels=NUM_CHANNELS
    ).to(device)
    num_params = count_pc_params(pc)
    num_trainable_params = count_trainable_parameters(pc)
    print(f"Number of parameters: {num_params}")

    # benchmark evaluation mode
    test_mu_t, test_sigma_t, test_mu_m, test_sigma_m = do_benchmarking(pc, data_loader, device, mode="test")
    # benchmark training mode
    train_mu_t, train_sigma_t, train_mu_m, train_sigma_m = do_benchmarking(pc, data_loader, device, mode="train")

    csv_row = {
        "rg": args.region_graph,
        "layer": args.layer,
        "k": args.k,
        "num-params": num_params,
        "num-trainable-params": num_trainable_params,
        "batch-size": args.batch_size,
        "test-time": test_mu_t,
        "test-time-std": test_sigma_t,
        "test-space": test_mu_m,
        "test-space-std": test_sigma_m,
        "train-time": train_mu_t,
        "train-time-std": train_sigma_t,
        "train-space": train_mu_m,
        "train-space-std": train_mu_t
    }

    df = pd.DataFrame.from_dict([csv_row])
    if os.path.exists(args.results_csv):
        df.to_csv(args.results_csv, mode="a", index=False, header=False)
    else:
        df.to_csv(args.results_csv, index=False)


if __name__ == "__main__":
    main()
