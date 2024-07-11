from typing import Optional

from probcirc.layers.sum_product import SharedCPLayer
from probcirc.reparams.leaf import ReparamIdentity
from probcirc.reparams.reparam import Reparameterization
from torch import Tensor
import torch
import torch.nn as nn

from probcirc.utils.type_aliases import ReparamFactory


class ScaledSharedCPLayer(SharedCPLayer):
    params_scale: Reparameterization

    def __init__(  # pylint: disable=too-many-arguments
            self,
            *,
            num_input_units: int,
            num_output_units: int,
            arity: int = 2,
            num_folds: int = 1,
            fold_mask: Optional[Tensor] = None,
            reparam: ReparamFactory = ReparamIdentity,
    ) -> None:
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            reparam=reparam
        )

        # Instantiate the fold-dependant scales
        self.params_scale = reparam(self._infer_shape("fo"), dim=-1, mask=None)
        for param in self.params_scale.parameters():
            nn.init.uniform_(param, 0.9, 1.1)

    def forward(self, x: Tensor) -> Tensor:
        y = super().forward(x)  # (F, K, *B)
        return y + torch.log(self.params_scale()).unsqueeze(dim=-1)
