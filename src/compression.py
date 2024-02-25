import argparse
import gc
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Literal
import sys
import os
import tensorly.decomposition
import torch

from tensorly import set_backend
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd(), "cirkit"))
sys.path.append(os.path.join(os.getcwd(), "src"))

from measures import eval_bpd
from cirkit.layers.input.exp_family import CategoricalLayer, BinomialLayer
from cirkit.region_graph.poon_domingos import PoonDomingos
from cirkit.region_graph.quad_tree import QuadTree
from cirkit.reparams.leaf import ReparamIdentity
from cirkit_extension.real_qt import RealQuadTree
from cirkit_extension.trees import TREE_DICT
from clt import tree2rg
from datasets import load_dataset
from cirkit_extension.tensorized_circuit import TensorizedPC
from cirkit.layers.sum import SumLayer
from cirkit.layers.sum_product import UncollapsedCPLayer, TuckerLayer


@dataclass
class _ArgsNamespace(argparse.Namespace):
    seed: int = 42
    gpu: int = 0
    tucker_model_path: str = ""
    save_model_dir: str = ""
    dataset: str = "mnist"
    batch_size: int = 128
    rg: str = "QG"
    rank: int = 1
    input_type: str = "cat"
    progressbar: bool = False


def process_args() -> _ArgsNamespace:
    """Process command line arguments.

    Returns:
        ArgsNamespace: Parsed args.
    """



def copy_uncompressible_params(orig_model: TensorizedPC, compressed_model: TensorizedPC):
    """Copies parameters of input and Sum layers of orig_model into compressed_model.
    Note: the two circuits must have the same structure

    Args:
        orig_model (TensorizedPC): _description_
        compressed_model (TensorizedPC): _description_
    """

    # copy input layer
    compressed_model.input_layer.load_state_dict(orig_model.input_layer.state_dict())

    # copy inner (sum) layers
    for n, layer in enumerate(orig_model.inner_layers):
        compressed_layer = compressed_model.inner_layers[n]
        if type(layer) == SumLayer:
            assert type(compressed_layer) == SumLayer
            assert torch.all(compressed_layer.params() > 0)
            # copy the parameters and buffers
            compressed_layer.load_state_dict(layer.state_dict())


def local_compression(orig_model: TensorizedPC, compressed_model: TensorizedPC,  rank: int):

    with torch.no_grad():

        for n, layer in enumerate(orig_model.inner_layers):
            if type(layer) == TuckerLayer:
                compressed_layer = compressed_model.inner_layers[n]
                assert type(compressed_layer) == UncollapsedCPLayer
                orig_tensor = layer.params()

                with torch.cuda.device_of(orig_tensor):
                    assert type(compressed_layer) == UncollapsedCPLayer
                    a, b, c = cp_slice_decomposition(orig_tensor, rank=rank)

                    new_params_in = torch.cat((a.unsqueeze(dim=1), b.unsqueeze(dim=1)), dim=1)
                    c = c.permute(dims=(0, 2, 1))
                    print(compressed_layer.params_in().shape)
                    print(new_params_in.shape)
                    assert compressed_layer.params_in().shape == new_params_in.shape
                    assert compressed_layer.params_out().shape == c.shape

                    compressed_layer.params_in().data = new_params_in
                    compressed_layer.params_out().data = c

                assert torch.all(compressed_layer.params_in() >= 0)
                assert torch.all(compressed_layer.params_out() >= 0)


# TODO: edit this
def cp_slice_decomposition(tensor: torch.Tensor, rank: int, progressbar: bool = False):
    """
    Given a (f, i, j, k) tensor, decomposes each (f, :, :, :) slice into (i, r), (j, r) and (k, r) matrices,
    then stack these matrices over the first dimensions, returning (f, i, r), (f, j, r) and (f, k, r)
    :param tensor: tensor to slice-decompose
    :param rank: rank of each decomposition
    :param progressbar: whether print the progress bar or not
    :return: (:, i, r), (:, j, r) and (:, k, r)
    """

    assert rank >= 1
    set_backend("pytorch")
    # TODO: change this
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # Note: possible alternative is NN_PARAFAC_HALS
    dec = tensorly.decomposition.non_negative_parafac  # (rank = r)

    with torch.no_grad():
        with torch.cuda.device_of(tensor.data):
            assert len(tensor.shape) == 4
            num_folds, i, j, k = tensor.shape


            cp_a = torch.empty(size=(num_folds, i, rank))
            cp_b = torch.empty(size=(num_folds, j, rank))
            cp_c = torch.empty(size=(num_folds, k, rank))

            cp_res = []
            errors = []

            pbar = range(num_folds)
            if progressbar:
                pbar = tqdm(iterable=pbar, total=len(pbar), unit="steps", ascii=" ▖▘▝▗▚▞█", ncols=120)

            for fold in pbar:
                (coeff, matrices), err_val = dec(
                    tensor[fold, :, :, :], rank=rank, return_errors=True
                )
                cp_res.append((coeff, matrices))
                # Note: Errors for now are not returned, but might be useful in the future
                errors.append(err_val[-1].item())

                for m in matrices:
                    if torch.isnan(m).any():
                        assert not torch.isnan(m).any()
                    if torch.isinf(m).any():
                        assert not torch.isinf(m).any()

                # TODO: check if this is necessary
                gc.collect()

            for node, (coeff, matrices) in enumerate(cp_res):
                cp_a[node, :, :] = matrices[0]
                cp_b[node, :, :] = matrices[1]
                cp_c[node, :, :] = matrices[2]

    print(f"Decomposed tensor of shape {tensor.shape}")
    return cp_a, cp_b, cp_c


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument("--gpu", type=int, help="Device on which run the benchmark")
    parser.add_argument("--tucker-model-path", type=str, help="Path of the tucker model to compress")
    parser.add_argument("--save-model-dir", type=str, help="Dir where to save the compressed model")
    parser.add_argument("--dataset", type=str, help="Dataset for the experiment")
    parser.add_argument("--rg", type=str, help="Region graph: 'PD', 'QG', 'QT' or 'RQT'")
    parser.add_argument("--rank", type=int, help="rank of the compressed model")
    parser.add_argument("--input-type", type=str, help="'bin' or 'cat'")
    parser.add_argument("--progressbar", type=bool, help="Print the progress bar")
    args = parser.parse_args(namespace=_ArgsNamespace())

    # get device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu"

    # load model to compress
    tucker_pc: TensorizedPC = torch.load(args.tucker_model_path).to(device)

    # create RG: todo remove duplication of code
    if args.dataset in ["mnist", "fashion_mnist"]:
        image_size = 28
    elif args.dataset == "celeba":
        image_size = 64
    else:
        raise AssertionError("Unknown dataset")

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

    # TODO: remove duplication of code
    INPUT_TYPES = {"cat": CategoricalLayer, "bin": BinomialLayer}

    efamily_kwargs = {
        'cat': {'num_categories': 256},
        'bin': {'n': 256}
    }[args.input_type]

    # create UncollapsedCPLayer model
    # TODO: edit efamily_cls and efamily_kwargs
    cp_pc = TensorizedPC.from_region_graph(
        rg=rg,
        layer_cls=UncollapsedCPLayer,
        layer_kwargs={"rank": args.rank},
        efamily_cls=INPUT_TYPES[args.input_type],
        efamily_kwargs=efamily_kwargs,
        num_inner_units=tucker_pc.inner_layers[0].num_output_units,
        num_input_units=tucker_pc.inner_layers[0].num_input_units,
        num_channels=tucker_pc.input_layer.num_channels,
        reparam=ReparamIdentity,
    ).to(device)
    copy_uncompressible_params(tucker_pc, cp_pc)
    local_compression(tucker_pc, cp_pc, rank=args.rank)

    # save model
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    torch.save(cp_pc, os.path.join(args.save_model_dir, f"rank_{args.rank}.mdl"))

    # evaluate test bpd
    _, _, test = load_dataset(args.dataset)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    print(f"Test bpd: {eval_bpd(cp_pc, test_loader, device)}")
