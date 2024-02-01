from typing import List

import numpy as np
import torch

from cirkit.new.model.functional import integrate
from cirkit.new.model.tensorized_circuit import TensorizedCircuit


def log_likelihoods(outputs, labels=None):
    """Compute the likelihood of EinsumNetwork."""
    if labels is None:
        num_dist = outputs.shape[-1]
        if num_dist == 1:
            lls = outputs
        else:
            num_dist = torch.tensor(float(num_dist), device=outputs.device)
            lls = torch.logsumexp(outputs - torch.log(num_dist), -1)
    else:
        lls = outputs.gather(-1, labels.unsqueeze(-1))
    return lls


def eval_loglikelihood_batched(
    pc: TensorizedCircuit, x: torch.Tensor, labels=None, batch_size=100
):
    """Computes log-likelihood in batched way."""
    with torch.no_grad():

        pc_pf = integrate(pc)
        idx_batches = torch.arange(
            0, x.shape[0], dtype=torch.int64, device=x.device
        ).split(batch_size)
        ll_total = 0.0
        for batch_count, idx in enumerate(idx_batches):
            batch_x = x[idx, :]
            if labels is not None:
                batch_labels = labels[idx]
            else:
                batch_labels = None

            outputs = pc(batch_x)
            ll_sample = log_likelihoods(outputs, batch_labels)
            ll_total += (ll_sample - pc_pf()).sum().item()
        return ll_total


def get_outputs_batched(pc: TensorizedCircuit, x: torch.Tensor, batch_size=100):

    with torch.no_grad():
        pc_pf = integrate(pc)
        output_list: List[torch.Tensor] = []  # shape x.shape[0], 1
        idx_batches = torch.arange(
            0, x.shape[0], dtype=torch.int64, device=x.device
        ).split(batch_size)

        for batch_count, idx in enumerate(idx_batches):
            batch_x = x[idx, :]

            batch_output = log_likelihoods(pc(batch_x))
            batch_output = batch_output - pc_pf()

            output_list.append(batch_output)

        output = torch.cat(output_list, dim=0)
        return output


def eval_bpd(pc: TensorizedCircuit, x: torch.Tensor) -> float:
    """
    Note: if ll is None then is computed, otherwise it is assumed that
    it has already been divided by the number of examples
    :param einet:
    :param x:
    :param ll:
    :return:
    """
    ll: float = eval_loglikelihood_batched(pc, x) / x.shape[0]
    return -ll / (np.log(2) * pc.num_variables)  # TODO: check this


def bpd_from_ll(pc: TensorizedCircuit, ll: float) -> float:
    return -ll / (np.log(2) * pc.num_variables)
