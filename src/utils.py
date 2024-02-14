import datetime
import os
import random
import numpy as np
import torch


from cirkit.models.tensorized_circuit import TensorizedPC


def load_model(path: str, device="cpu") -> TensorizedPC:
    return torch.load(path, map_location=device)


def get_date_time_str() -> str:
    now = datetime.datetime.now()
    return now.strftime("%d_%m_%Y_%H_%M_%S")


def init_random_seeds(seed: int = 42):
    """Seed all random generators and enforce deterministic algorithms to \
        guarantee reproducible results (may limit performance).

    Args:
        seed (int): The seed shared by all RNGs.
    """
    seed = seed % 2**32  # some only accept 32bit seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def num_of_params(pc: TensorizedPC) -> int:
    num_param = sum(p.numel() for p in pc.input_layer.parameters())
    for layer in pc.inner_layers:
        num_param += sum(p.numel() for p in layer.parameters())
    return num_param


def count_pc_params(pc: TensorizedPC) -> int:
    num_param = pc.input_layer.params.param.numel()
    for layer in pc.inner_layers:
        num_param += layer.params_in.param.numel()
    return num_param


def get_pc_device(pc: TensorizedPC) -> torch.DeviceObjType:
    for par in pc.input_layer.parameters():
        return par.device


def check_validity_params(pc: TensorizedPC):

    for p in pc.input_layer.parameters():
        if torch.isnan(p.grad).any():
            raise AssertionError(f"NaN grad in input layer")
        elif torch.isinf(p.grad).any():
            raise AssertionError(f"Inf grad in input layer")

    for num, layer in enumerate(pc.inner_layers):
        for p in layer.parameters():
            if torch.isnan(p.grad).any():
                raise AssertionError(f"NaN grad in {num}, {type(layer)}")
            elif torch.isinf(p.grad).any():
                raise AssertionError(f"Inf grad in {num}, {type(layer)}")


def param_to_buffer(module):
    """Turns all parameters of a module into buffers."""
    modules = module.modules()
    module = next(modules)
    for name, param in module.named_parameters(recurse=False):
        delattr(module, name)  # Unregister parameter
        module.register_buffer(name, param.data)
    for module in modules:
        param_to_buffer(module)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
