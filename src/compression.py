import gc
import warnings
from typing import Any, Dict, Literal
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "cirkit"))
sys.path.append(os.path.join(os.getcwd(), "src"))

import tensorly.decomposition
import torch
from utils import get_date_time_str
from tensorly import set_backend

from cirkit.models.tensorized_circuit import TensorizedPC
from cirkit.layers.sum import SumLayer
from cirkit.layers.sum_product import UncollapsedCPLayer, TuckerLayer




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


def local_compression(
    orig_model: TensorizedPC, compressed_model: TensorizedPC, mode: Literal["slice", "full"] = "slice"
):
    assert mode == "slice" or mode == "full"

    # TODO: check same structure
    with torch.no_grad():

        for n, layer in enumerate(orig_model.inner_layers):
            if type(layer) == TuckerLayer:
                compressed_layer = compressed_model.inner_layers[n]
                assert type(compressed_layer) == UncollapsedCPLayer
                orig_tensor = layer.params()

                a = compressed_layer.params_in() # TODO find a way to select only a slice of a tensor
                b = compressed_layer.params_in()
                c = compressed_layer.params_out()
                with torch.cuda.device_of(orig_tensor):
                    if mode == "full":
                        raise NotImplementedError
                        assert type(compressed_layer) == CPSharedEinsumLayer
                        d = lora_params_dict["cp_d"]
                        cp_full_tensor_decomposition(orig_tensor, a, b, c, d)
                    else:
                        assert type(compressed_layer) == UncollapsedCPLayer
                        a, b, c = cp_slice_decomposition(orig_tensor)

                assert torch.all(compressed_layer.params_in() >= 0)
                assert torch.all(compressed_layer.params_out() >= 0)


# TODO: edit this
def cp_slice_decomposition(tensor: torch.Tensor, rank: int):
    """
    Given a (f, i, j, k) tensor, decomposes each (f, :, :, :) slice into (i, r), (j, r) and (k, r) matrices,
    then stack these matrices over the first dimensions, returning (f, i, r), (f, j, r) and (f, k, r)
    :param tensor: tensor to slice-decompose
    :return: (:, i, r), (:, j, r) and (:, k, r)
    """

    assert rank >= 1
    set_backend("pytorch")
    # TODO: change this
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # TODO: edit in future
    method = "CP_NN"
    if method == "CP_NN_HALS":
        dec = tensorly.decomposition.non_negative_parafac_hals  # (rank=r)
    elif method == "CP_NN":
        dec = tensorly.decomposition.non_negative_parafac  # (rank = r)
    else:
        raise AssertionError("unknown method for decomposition")

    with torch.no_grad():
        with torch.cuda.device_of(tensor.data):
            assert len(tensor.shape) == 4
            num_folds, i, j, k = tensor.shape

            cp_a = torch.empty(size=(num_folds, i, rank))
            cp_b = torch.empty(size=(num_folds, j, rank))
            cp_c = torch.empty(size=(num_folds, k, rank))

            cp_res = []
            errors = []

            for fold in range(num_folds):
                (coeff, matrices), err_val = dec(
                    tensor[fold, :, :, :], rank=rank, return_errors=True
                )
                cp_res.append((coeff, matrices))
                errors.append(err_val[-1].item())

                for m in matrices:
                    if torch.isnan(m).any():
                        assert not torch.isnan(m).any()
                    if torch.isinf(m).any():
                        assert not torch.isinf(m).any()

                if num_nodes % 32 == 31:
                    print("-", end="")
                gc.collect()  # TODO: check if this is necessary

            errors_mean = np.mean(errors)
            errors_std = np.std(errors)

            for node, (coeff, matrices) in enumerate(cp_res):
                cp_a[:, :, node] = matrices[0]
                cp_b[:, :, node] = matrices[1]
                cp_c[:, :, node] = matrices[2]

    print(f"Decomposed tensor of shape {tensor.shape}")

    return errors_mean, errors_std


def cp_full_tensor_decomposition(
    tensor: torch.Tensor,
    cp_a: torch.Tensor,
    cp_b: torch.Tensor,
    cp_c: torch.Tensor,
    cp_d: torch.Tensor,
):
    raise NotImplementedError
    """
    Given a (i, j, k, p) tensor, decomposes it into (i, r), (j, r), (k, r) and (p, r) matrices
    :param tensor: tensor to slice-decompose
    :param cp_a: tensor "container" for the first slice decompositions components
    :param cp_b: container for the second component
    :param cp_c: container for the third component
    :param cp_d: container for the fourth component
    :return:
    """

    set_backend("pytorch")
    # TODO: change this
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # TODO: edit in future
    method = "CP_NN"
    if method == "CP_NN_HALS":
        dec = tensorly.decomposition.non_negative_parafac_hals  # (rank=r)
    elif method == "CP_NN":
        dec = tensorly.decomposition.non_negative_parafac  # (rank = r)
    else:
        raise AssertionError("unknown method for decomposition")

    with torch.no_grad():
        assert len(tensor.shape) == 4
        i, j, k, p = tensor.shape

        r = cp_a.shape[1]

        assert len(cp_a.shape) == 2 and cp_a.shape[0] == i
        assert len(cp_b.shape) == 2 and cp_b.shape[0] == j
        assert len(cp_c.shape) == 2 and cp_c.shape[0] == k
        assert len(cp_d.shape) == 2 and cp_d.shape[0] == p

        if k == 1:  # case 1 output sum
            if r >= i / 2:
                warnings.warn(
                    f"Decomposition of a ({i, j, k}) tensor lead to {(i + j + k) * r} parameters"
                )

        (coeff, matrices), err_val = dec(
            tensor, init="random", rank=r, return_errors=True
        )
        gc.collect()  # TODO: check if this is necessary

        for m in matrices:
            if torch.isnan(m).any():
                assert not torch.isnan(m).any()
            if torch.isinf(m).any():
                assert not torch.isinf(m).any()

        errors_mean = err_val
        errors_std = 0

        cp_a[:, :] = matrices[0]
        cp_b[:, :] = matrices[1]
        cp_c[:, :] = matrices[2]
        cp_d[:, :] = matrices[3]

    print(f"Decomposed tensor of shape {tensor.shape}")

    return errors_mean, errors_std
