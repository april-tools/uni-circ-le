import gc
import warnings
from typing import Any, Dict

import tensorly.decomposition
import torch
from experiments.exp_utils import get_date_time_str
from LoRaEinNet.FactorizedLeafLayer import FactorizedLeafLayer
from LoRaEinNet.LoRaEinsumLayer import (
    CPEinsumLayer,
    CPSharedEinsumLayer,
    EinsumMixingLayer,
    HCPTEinsumLayer,
    HCPTLoLoEinsumLayer,
    HCPTLoLoSharedEinsumLayer,
    HCPTSharedEinsumLayer,
    RescalEinsumLayer,
)
from tensorly import set_backend


def copy_uncompressible_params(orig_model: EinsumNetwork, lora_model: LoRaEinNetwork):
    for n, layer in enumerate(orig_model.einet_layers):
        lora_layer = lora_model.einet_layers[n]
        if type(layer) == RescalEinsumLayer:
            assert torch.all(layer.params > 0)
            assert type(lora_layer) in [
                CPEinsumLayer,
                CPSharedEinsumLayer,
                RescalEinsumLayer,
            ]
            if type(lora_layer) == RescalEinsumLayer:  # TODO: bug here
                lora_layer.load_state_dict(layer.state_dict())
                lora_layer.clamp_params(all=True)
        elif type(layer) == EinsumMixingLayer:
            assert type(lora_layer) == EinsumMixingLayer
            assert torch.all(layer.params > 0)
            # copy the parameters and buffers
            lora_layer.load_state_dict(layer.state_dict())
            lora_layer.clamp_params(all=True)
        elif type(layer) == FactorizedLeafLayer:
            assert type(lora_layer) == FactorizedLeafLayer
            # copy the parameters and buffers
            lora_layer.load_state_dict(layer.state_dict())
        else:
            raise AssertionError(f"You should not be here, layer is {type(layer)}")


def copy_uncompressible_params_lora(
    orig_lora_model: LoRaEinNetwork, lolo_model: LoRaEinNetwork
):
    for n, layer in enumerate(orig_lora_model.einet_layers):
        lolo_layer = lolo_model.einet_layers[n]
        if type(layer) == HCPTEinsumLayer:
            assert type(lolo_layer) in [
                HCPTLoLoEinsumLayer,
                HCPTLoLoSharedEinsumLayer,
                HCPTEinsumLayer,
            ]
            if type(lolo_layer) == HCPTEinsumLayer:
                lolo_layer.load_state_dict(layer.state_dict())
        elif type(layer) == EinsumMixingLayer:
            assert type(lolo_layer) == EinsumMixingLayer
            lolo_layer.load_state_dict(
                layer.state_dict()
            )  # copy parameters and buffers
        elif type(layer) == FactorizedLeafLayer:
            assert type(lolo_layer) == FactorizedLeafLayer
            lolo_layer.load_state_dict(
                layer.state_dict()
            )  # copy parameters and buffers
        else:
            raise AssertionError(f"You should not be here, layer is {type(layer)}")

    # CHECK
    for n, lolo_layer in enumerate(lolo_model.einet_layers):
        layer = orig_lora_model.einet_layers[n]
        if type(layer) == HCPTEinsumLayer and type(lolo_layer) in [
            HCPTLoLoEinsumLayer,
            HCPTLoLoSharedEinsumLayer,
        ]:
            continue
        if type(lolo_layer) in [FactorizedLeafLayer, EinsumMixingLayer]:
            for par in lolo_layer.state_dict():
                assert torch.all(
                    torch.eq(lolo_layer.state_dict()[par], layer.state_dict()[par])
                )
        else:
            for par in lolo_layer.get_params_dict():
                assert torch.all(
                    torch.eq(
                        lolo_layer.get_params_dict()[par], layer.get_params_dict()[par]
                    )
                )


def local_compression(
    orig_model: LoRaEinNetwork, lora_model: LoRaEinNetwork, mode="slice"
):
    """

    :param orig_model:
    :param lora_model:
    :param mode: "slice" or "full"
    :return:
    """
    assert mode == "slice" or mode == "full"

    # TODO: check same structure
    with torch.no_grad():

        for n, layer in enumerate(orig_model.einet_layers):
            if type(layer) == RescalEinsumLayer:
                lora_layer = lora_model.einet_layers[n]
                if type(lora_layer) == RescalEinsumLayer:
                    print("Skip")
                    continue
                assert type(lora_layer) in [CPEinsumLayer, CPSharedEinsumLayer]
                orig_tensor = layer.get_params_dict()["params"]

                lora_params_dict = lora_layer.get_params_dict()
                a = lora_params_dict["cp_a"]
                b = lora_params_dict["cp_b"]
                c = lora_params_dict["cp_c"]

                with torch.cuda.device_of(orig_tensor):
                    if mode == "full":
                        assert type(lora_layer) == CPSharedEinsumLayer
                        d = lora_params_dict["cp_d"]
                        cp_full_tensor_decomposition(orig_tensor, a, b, c, d)
                    else:
                        assert type(lora_layer) == CPEinsumLayer
                        cp_slice_decomposition(orig_tensor, a, b, c)

                lora_layer.clamp_params(all=True)


def local_compression_HCLT(
    orig_model: LoRaEinNetwork, lolo_model: LoRaEinNetwork, mode="slice"
):
    """

    :param orig_model:
    :param lolo_model:
    :param mode: "slice" or "full"
    :return:
    """
    assert mode == "slice" or mode == "full"

    # TODO: check same structure
    with torch.no_grad():

        for n, layer in enumerate(orig_model.einet_layers):
            if type(layer) == HCPTEinsumLayer:
                lolo_layer = lolo_model.einet_layers[n]
                if type(lolo_layer) == HCPTEinsumLayer:
                    continue

                orig_tensor_a = layer.get_params_dict()["cp_a"]
                orig_tensor_b = layer.get_params_dict()["cp_b"]

                if orig_tensor_a.shape[0] != orig_tensor_a.shape[1]:
                    print(f"Tensor shape: {orig_tensor_a.shape}, SKIPPED")
                    continue

                lora_params_dict = lolo_layer.get_params_dict()
                a_1 = lora_params_dict["cp_a1"]
                a_2 = lora_params_dict["cp_a2"]
                b_1 = lora_params_dict["cp_b1"]
                b_2 = lora_params_dict["cp_b2"]

                with torch.cuda.device_of(orig_tensor_a):
                    if mode == "full":
                        d_1 = lora_params_dict["cp_d1"]
                        d_2 = lora_params_dict["cp_d2"]
                        lolo_shared_decomposition(orig_tensor_a, a_1, a_2, d_1)
                        lolo_shared_decomposition(orig_tensor_b, b_1, b_2, d_2)
                    else:
                        cp_nmf(orig_tensor_a, a_1, a_2)
                        cp_nmf(orig_tensor_b, b_1, b_2)

                lolo_layer.clamp_params(all=True)


def cp_nmf(tensor: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
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
        assert len(tensor.shape) == 3
        k1, k2, p = tensor.shape

        r = a.shape[1]

        assert len(a.shape) == 3 and a.shape[0] == k1 and a.shape[2] == p
        assert (
            len(b.shape) == 3
            and b.shape[0] == r
            and b.shape[1] == k2
            and b.shape[2] == p
        )

        num_nodes = p

        cp_res = []
        errors = []

        for node in range(num_nodes):
            (coeff, matrices), err_val = dec(
                tensor[:, :, node], rank=r, return_errors=True
            )

            for m in matrices:
                if torch.isnan(m).any():
                    assert not torch.isnan(m).any()
                if torch.isinf(m).any():
                    assert not torch.isinf(m).any()

            cp_res.append((coeff, matrices))
            errors.append(err_val[-1].item())
            gc.collect()  # TODO: check if this is necessary

        errors_mean = np.mean(errors)
        errors_std = np.std(errors)

        for node, (coeff, matrices) in enumerate(
            cp_res
        ):  # TODO align with the other code
            a[:, :, node] = matrices[0]
            b[:, :, node] = torch.t(matrices[1])

    print(f"Decomposed tensor of shape {tensor.shape}")

    return errors_mean, errors_std


def lolo_shared_decomposition(
    tensor: torch.Tensor, a: torch.Tensor, b: torch.Tensor, d: torch.Tensor
):
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
        assert len(tensor.shape) == 3
        k_1, k_2, p = tensor.shape

        r = a.shape[1]

        assert len(a.shape) == 2 and a.shape[0] == k_1
        assert len(b.shape) == 2 and b.shape[1] == k_2
        assert len(d.shape) == 2 and d.shape[0] == p

        (coeff, matrices), err_val = dec(tensor, rank=r, return_errors=True)
        gc.collect()  # TODO: check if this is necessary

        errors_mean = err_val
        errors_std = 0

        a[:, :] = matrices[0]
        b[:, :] = torch.t(matrices[1])
        d[:, :] = matrices[2]

    print(f"Decomposed tensor of shape {tensor.shape}")

    return errors_mean, errors_std


def cp_slice_decomposition(
    tensor: torch.Tensor, cp_a: torch.Tensor, cp_b: torch.Tensor, cp_c: torch.Tensor
):
    """
    Given a (i, j, k, p) tensor, decomposes each (i, j, k, 1) slice into (i, r), (j, r) and (k, r) matrices,
    then stack these matrices over the last dimension of cp_a (i, r, p), cp_b (j, r, p) and cp_c (k, r, p)
    :param tensor: tensor to slice-decompose
    :param cp_a: tensor "container" for the first slice decompositions components
    :param cp_b: container for the second component
    :param cp_c: container for the third component
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
        with torch.cuda.device_of(tensor.data):
            assert len(tensor.shape) == 4
            i, j, k, p = tensor.shape

            r = cp_a.shape[1]

            assert len(cp_a.shape) == 3 and cp_a.shape[0] == i and cp_a.shape[2] == p
            assert len(cp_b.shape) == 3 and cp_b.shape[0] == j and cp_b.shape[2] == p
            assert len(cp_c.shape) == 3 and cp_c.shape[0] == k and cp_c.shape[2] == p

            if k == 1:  # case 1 output sum
                if r >= i / 2:
                    warnings.warn(
                        f"Decomposition of a ({i, j, k}) tensor lead to {(i + j + k) * r} parameters"
                    )

            num_nodes = p

            cp_res = []
            errors = []

            for node in range(num_nodes):
                (coeff, matrices), err_val = dec(
                    tensor[:, :, :, node], rank=r, return_errors=True
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
