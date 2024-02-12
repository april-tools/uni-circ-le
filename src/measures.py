from typing import List

import numpy as np
import torch

from cirkit.models.functional import integrate
from cirkit.models.tensorized_circuit import TensorizedPC
from torch.utils.data import DataLoader


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
    pc: TensorizedPC, data_loader: DataLoader, labels=None, batch_size=100, device: str = 'cpu'
):
    """Computes log-likelihood in batched way."""
    with torch.no_grad():

        pc_pf = integrate(pc)
        ll_total = 0.0
        for batch in data_loader:
            batch = batch.to(device)
            outputs = pc(batch)
            ll_sample = log_likelihoods(outputs, labels=None)
            ll_total += (ll_sample - pc_pf(batch)).sum().item()
        return ll_total / len(data_loader.dataset)


def get_outputs_batched(pc: TensorizedPC, x: torch.Tensor, batch_size=100):

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


def eval_bpd(pc: TensorizedPC, data_loader: DataLoader, device: str = 'cpu') -> float:
    """
    Note: if ll is None then is computed, otherwise it is assumed that
    it has already been divided by the number of examples
    """
    ll: float = eval_loglikelihood_batched(pc, data_loader, device=device)
    return ll2bpd(ll, pc.num_vars)


def ll2bpd(ll: float, num_vars: int) -> float:
    return -ll / (np.log(2) * num_vars)
