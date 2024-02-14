from cirkit.reparams.leaf import ReparamLeaf
import torch


# todo num_sums needs to be passed in __init__
"""
import numpy as np

class ReparamExpTemp(ReparamLeaf):
    def forward(self) -> torch.Tensor:
        return torch.exp(self.param / np.sqrt(args.num_sums))


class ReparamSoftmaxTemp(ReparamLeaf):
    def forward(self) -> torch.Tensor:
        param = self.param if self.log_mask is None else self.param + self.log_mask
        param = self._unflatten_dims(
            torch.softmax(self._flatten_dims(param) / np.sqrt(args.num_sums), dim=self.dims[0]))
        return torch.nan_to_num(param, nan=1)
"""


class ReparamReLU(ReparamLeaf):
    eps = torch.finfo(torch.get_default_dtype()).tiny

    def forward(self) -> torch.Tensor:
        return torch.clamp(self.param, min=ReparamReLU.eps)


class ReparamSoftplus(ReparamLeaf):
    def forward(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.param)
