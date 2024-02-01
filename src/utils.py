import datetime
import os
import random
from collections import Counter

import numpy as np
import torch
from emnist import extract_test_samples, extract_training_samples

import datasets
from cirkit.new.model.tensorized_circuit import TensorizedCircuit

DEBD = [
    "ad",
    "accidents",
    "baudio",
    "bbc",
    "bnetflix",
    "book",
    "c20ng",
    "cr52",
    "cwebkb",
    "dna",
    "jester",
    "kdd",
    "kosarek",
    "moviereview",
    "msnbc",
    "msweb",
    "nltcs",
    "plants",
    "pumsb_star",
    "tmovie",
    "tretail",
    "voting",
]

MNIST = ["mnist", "fashion_mnist", "balanced", "byclass", "letters", "e_mnist"]


def load_model(path: str, device="cpu") -> TensorizedCircuit:
    return torch.load(path, map_location=device)


def load_dataset(name: str, device):
    """
    Load a dataset into a device
    :param name: dataset name (one of DEBD or MNIST datasets)
    :param device: device to load the dataset into
    :return: train_x, valid_x, test_x
    """
    if torch.cuda.is_available():
        torch.random.fork_rng(devices=[device])
    else:
        torch.random.fork_rng()

    if name in DEBD:
        train_x, test_x, valid_x = datasets.load_debd(name, dtype="float32")
    elif name in MNIST:
        if name in ["fashion_mnist", "mnist"]:
            if name == "fashion_mnist":
                (
                    train_x,
                    train_labels,
                    test_x,
                    test_labels,
                ) = datasets.load_fashion_mnist()
            else:
                train_x, train_labels, test_x, test_labels = datasets.load_mnist()

            valid_x = train_x[-3000:, :]
            train_x = train_x[:-3000, :]

            # print(Counter(train_labels[-3000:]))

        elif name in ["balanced", "byclass", "letters", "e_mnist"]:
            if name == "e_mnist":
                name = "mnist"
            train_x, train_labels = extract_training_samples(name)
            test_x, test_labels = extract_test_samples(name)

            train_x = train_x.reshape(-1, 784)
            test_x = test_x.reshape(-1, 784)

            # TODO: fix here
            percentage_5_train = int(train_x.shape[0] / 20)
            valid_x = train_x[-percentage_5_train:, :]
            train_x = train_x[:-percentage_5_train, :]

            # print(Counter(train_labels[-percentage_5_train:]))
        else:
            raise AssertionError("Inconsistent mnist value ?!")

    else:
        raise AssertionError("Invalid dataset name")

    train_x = torch.from_numpy(train_x).to(torch.device(device))
    valid_x = torch.from_numpy(valid_x).to(torch.device(device))
    test_x = torch.from_numpy(test_x).to(torch.device(device))

    return train_x, valid_x, test_x


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


def num_of_params(pc: TensorizedCircuit) -> int:
    num_param = sum(p.numel() for p in pc.input_layer.parameters())
    for layer in pc.inner_layers:
        num_param += sum(p.numel() for p in layer.parameters())

    return num_param


def get_pc_device(pc: TensorizedCircuit) -> torch.DeviceObjType:
    for par in pc.input_layer.parameters():
        return par.device


def check_validity_params(pc: TensorizedCircuit):

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
