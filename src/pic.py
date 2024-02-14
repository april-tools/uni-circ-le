from typing import Optional
import torch.nn as nn
import numpy as np
import torch


def zw_quadrature(
        mode: str,
        nip: int,
        a: Optional[float] = -1,
        b: Optional[float] = 1,
        log_weight: Optional[bool] = False,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[str] = 'cpu'
):
    if mode == 'leggauss':
        z, w = np.polynomial.legendre.leggauss(nip)
        z = (b - a) * (z + 1) / 2 + a
        w = w * (b - a) / 2
    elif mode == 'midpoint':
        z = np.linspace(a, b, num=nip + 1)
        z = (z[:-1] + z[1:]) / 2
        w = np.full_like(z, (b - a) / nip)
    elif mode == 'trapezoidal':
        z = np.linspace(a, b, num=nip)
        w = np.full((nip,), (b - a) / (nip - 1))
        w[0] = w[-1] = 0.5 * (b - a) / (nip - 1)
    elif mode == 'simpson':
        assert nip % 2 == 1, 'Number of integration points must be odd'
        z = np.linspace(a, b, num=nip)
        w = np.concatenate([np.ones(1), np.tile(np.array([4, 2]), nip // 2 - 1), np.array([4, 1])])
        w = ((b - a) / (nip - 1)) / 3 * w
    elif mode == 'hermgauss':
        # https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
        z, w = np.polynomial.hermite.hermgauss(nip)
    else:
        raise NotImplementedError('Integration mode not implemented.')
    z, w = torch.tensor(z, dtype=dtype), torch.tensor(w, dtype=dtype)
    w = w.log() if log_weight else w
    return z.to(device), w.to(device)


class FourierLayer(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            sigma: Optional[float] = 1.0,
            flatten01: Optional[bool] = False,
            learnable: Optional[bool] = False
    ):
        super(FourierLayer, self).__init__()
        assert out_features % 2 == 0, 'Number of output features must be even.'
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        self.flatten01 = flatten01
        if learnable:
            self.coeff = nn.Parameter(torch.normal(0.0, sigma, (in_features, out_features // 2)))
        else:
            self.register_buffer('coeff', torch.normal(0.0, sigma, (in_features, out_features // 2)))

    def forward(self, x: torch.Tensor):
        x_proj = 2 * torch.pi * x @ self.coeff
        x_ff = torch.cat([x_proj.cos(), x_proj.sin()], dim=-1).transpose(-2, -1)
        return x_ff.flatten(0, 1) if self.flatten01 else x_ff

    def extra_repr(self) -> str:
        return '{}, {}, sigma={}'.format(self.in_features, self.out_features, self.sigma)


class PIC(nn.Module):

    def __init__(
        self,
        n_inner_layers: int,
        inner_layer_type: str,
        n_input_layers: int,
        input_layer_type: str,
        net_dim: Optional[int] = 64,
        bias: Optional[bool] = False,
        n_categories: Optional[int] = None,
        sigma: Optional[float] = 1.0,
        learn_ff: Optional[bool] = False,
        multi_heads_inner_net: Optional[bool] = False,
        multi_heads_input_net: Optional[bool] = False,
        single_input_net: Optional[bool] = False
    ):
        super().__init__()
        if single_input_net: assert not multi_heads_input_net
        assert inner_layer_type in ['cp', 'tucker']
        #  input_type_dict = {'bernoulli': 1, 'binomial': 1, 'categorical': n_categories, 'gaussian': 2}
        input_type_dict = {'categorical': n_categories}
        assert input_layer_type in input_type_dict.keys()

        self.n_inner_layers = n_inner_layers
        self.inner_layer_type = inner_layer_type
        self.n_input_layers = n_input_layers
        self.input_layer_type = input_layer_type

        self.net_input_dim_input_layers = n_input_layers
        self.net_dim = net_dim
        self.n_categories = n_categories  # if input_layer_type is categorical, this specifies the number of categories
        self.multi_heads_inner_net = multi_heads_inner_net
        self.multi_heads_input_net = multi_heads_input_net
        self.single_input_net = single_input_net

        # self.root_net = nn.Sequential(
        #     FourierLayer(1, net_dim, sigma, learnable=learn_ff),
        #     nn.Conv1d(net_dim, net_dim, 1, bias=False),
        #     nn.Tanh(),
        #     nn.Conv1d(net_dim, 1, 1, bias=False),
        #     nn.Softplus())

        input_dim = 2 if self.inner_layer_type == 'cp' else 3
        conv1_dim = 1 if multi_heads_inner_net else n_inner_layers
        conv2_dim = n_inner_layers
        self.inner_net = nn.Sequential(
            FourierLayer(input_dim, net_dim, sigma, learnable=learn_ff),
            nn.Conv1d(conv1_dim * net_dim, conv1_dim * net_dim, 1, groups=conv1_dim, bias=bias),
            nn.Tanh(),
            nn.Conv1d(conv2_dim * net_dim, conv2_dim, 1, groups=conv2_dim, bias=bias),
            nn.Softplus())

        input_dim = 1 if self.inner_layer_type == 'cp' else 2
        conv1_dim = 1 if multi_heads_input_net or single_input_net else n_input_layers
        conv2_dim = 1 if single_input_net else n_input_layers
        self.input_net = nn.Sequential(
            FourierLayer(input_dim, net_dim, sigma, learnable=learn_ff),
            nn.Conv1d(conv1_dim * net_dim, conv1_dim * net_dim, 1, groups=conv1_dim, bias=bias),
            nn.SiLU() if input_layer_type == 'gaussian' else nn.Tanh(),
            nn.Conv1d(conv2_dim * net_dim, conv2_dim * input_type_dict[input_layer_type], 1, groups=conv2_dim, bias=bias))

    def quad(
        self,
        z: torch.Tensor,
        log_w: torch.Tensor,
        n_chunks: Optional[int] = 1
    ):
        assert z.ndim == 1
        nip = z.size(0)  # number of integration points

        if self.inner_layer_type == 'cp':

            self.inner_net[0].flatten01 = False
            self.inner_net[1].groups = 1
            if self.multi_heads_inner_net: self.inner_net[-2].groups = 1
            z2d = torch.stack([z.repeat_interleave(nip), z.repeat(nip)]).t()
            # sum_logits = torch.eye(nip, device=z.device).log().unsqueeze(0).repeat(self.n_inner_layers, 1, 1)
            # sum_logits[0] = - self.root_net(z.unsqueeze(1))
            inner_logits = - torch.hstack([self.inner_net(chunk) for chunk in z2d.chunk(n_chunks, 1)]).view(-1, nip, nip)
            log_inner_param = (inner_logits - (inner_logits + log_w).logsumexp(-1, True)) + log_w
            inner_param = log_inner_param.exp().transpose(-1, -2)  # apparently, cirkit normalizes over the second last dim

            self.input_net[0].flatten01 = False
            self.input_net[1].groups = 1
            if self.multi_heads_input_net: self.input_net[-1].groups = 1
            input_logits = self.input_net(z.unsqueeze(1)).squeeze()
            if self.single_input_net: input_logits = input_logits.unsqueeze(0).expand(self.n_input_layers, -1, -1)
            input_param = input_logits.view(self.n_input_layers, self.n_categories, nip).transpose(1, 2).log_softmax(-1)

            return inner_param, input_param
        else:
            raise NotImplementedError('Tucker variant not ready yet')


def parameterize_pc(pc, sum_param, input_param):
    from cirkit.layers.sum_product import CollapsedCPLayer, TuckerLayer
    # from cirkit.layers.sum import SumLayer

    matrices_per_layer = []
    for layer in pc.inner_layers:
        if isinstance(layer, CollapsedCPLayer):
            matrices_per_layer.append(layer.params_in.param.size()[:2].numel())
        elif isinstance(layer, TuckerLayer):
            matrices_per_layer.append(layer.params.param.size(0))
        else:
            raise Exception('layer not supported: ', type(layer))

    if isinstance(layer, CollapsedCPLayer):
        sum_param_chunks = sum_param.split(matrices_per_layer, 0)
        for layer, chunk, in zip(pc.inner_layers[:-1], sum_param_chunks[:-1]):
            layer.params_in.param = chunk.view_as(layer.params_in.param)
        pc.inner_layers[-1].params_in.param = sum_param_chunks[-1][..., :1].view_as(pc.inner_layers[-1].params_in.param)
        pc.input_layer.params.param = input_param.unsqueeze(2)
    else:
        raise NotImplementedError('Tucker variant not ready yet')

"""
for layer in pc.inner_layers:
    if isinstance(layer, CollapsedCPLayer):
        print(type(layer), layer.params_in.param.shape)
    else:
        print(type(layer), layer.params.param.shape)
"""
