from torchvision.datasets import MNIST, FashionMNIST, EMNIST, CIFAR10, CelebA
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from typing import Optional, Callable
import numpy as np
import torch
import csv

from data.transforms import PIC2Tensor, UnsqueezeDim, Flatten, UnsupervisedDataset, Rgb2YccLossless, Rgb2YccLossy, Identity, TensorDataset, ToLong, Permute


DEBD_DATASETS = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail',
                 'pumsb_star', 'dna', 'kosarek', 'msweb', 'book', 'tmovie', 'binarized_mnist', 'cwebkb', 'cr52',
                 'c20ng', 'bbc', 'ad']

MNIST_DATASETS = ['mnist', 'fashion_mnist', 'emnist']

UCI_DATASETS = ['bsds300', 'gas', 'miniboone', 'power', 'hepmass']


def load_dataset(
    ds_name: str,
    root: Optional[str] = './data/',
    return_torch: Optional[bool] = True,
    split: Optional[str] = 'mnist',
    valid_split_percentage: Optional[float] = 0,
    transform=None,
    ycc: Optional[str] = 'none'
):
    if ds_name in DEBD_DATASETS:
        return load_debd(ds_name, return_torch)
    elif ds_name in UCI_DATASETS:
        return load_uci(ds_name, return_torch, root=root)
    elif ds_name in MNIST_DATASETS:
        return load_mnist_family_dataset(ds_name, split=split, root=root, valid_split_percentage=valid_split_percentage)
    elif 'cifar' in ds_name:
        return load_cifar10(root=root, transform=transform, valid_split_percentage=valid_split_percentage, ycc=ycc)
    elif ds_name == 'celeba':
        train = CelebADataset(root=root, split='train', ycc=ycc)
        valid = CelebADataset(root=root, split='valid', ycc=ycc)
        test = CelebADataset(root=root, split='test', ycc=ycc)
        return train, valid, test
    elif ds_name == 'imagenet32':
        return load_imagenet(transform, valid_split_percentage=valid_split_percentage, ycc=ycc, image_size=32)
    elif ds_name == 'imagenet64':
        return load_imagenet(transform, valid_split_percentage=valid_split_percentage, ycc=ycc, image_size=64)
    elif ds_name == 'gpt2':
        return load_gpt2()
    else:
        raise Exception('Dataset %s not found' % ds_name)


def load_debd(
    ds_name: str,
    return_torch: Optional[bool] = True,
    dtype: str = np.uint8
):
    assert ds_name in DEBD_DATASETS
    train_path = './data/debd/%s/%s.train.data' % (ds_name, ds_name)
    valid_path = './data/debd/%s/%s.valid.data' % (ds_name, ds_name)
    test_path = './data/debd/%s/%s.test.data' % (ds_name, ds_name)
    reader = csv.reader(open(train_path, 'r'), delimiter=',' if ds_name != 'binarized_mnist' else ' ')
    train = np.array(list(reader)).astype(dtype)
    reader = csv.reader(open(test_path, 'r'), delimiter=',' if ds_name != 'binarized_mnist' else ' ')
    test = np.array(list(reader)).astype(dtype)
    reader = csv.reader(open(valid_path, 'r'), delimiter=',' if ds_name != 'binarized_mnist' else ' ')
    valid = np.array(list(reader)).astype(dtype)
    if return_torch:
        train, valid, test = torch.tensor(train), torch.tensor(valid), torch.tensor(test)
    return train, valid, test


def load_uci(
    ds_name: str,
    return_torch: Optional[bool] = True,
    root: Optional[str] = "./data/",
    add_channel: Optional[bool] = True
):
    # https://github.com/conormdurkan/autoregressive-energy-machines/blob/master/pytorch/utils/uciutils.py
    assert ds_name in UCI_DATASETS
    train = np.load(root + 'UCI/%s/train.npy' % ds_name)
    valid = np.load(root + 'UCI/%s/valid.npy' % ds_name)
    test = np.load(root + 'UCI/%s/test.npy' % ds_name)
    if add_channel:
        train, valid, test = train[..., None], valid[..., None], test[..., None]
    if return_torch:
        train, valid, test = torch.Tensor(train), torch.Tensor(valid), torch.Tensor(test)
    return train, valid, test


def load_gpt2():
    # https://github.com/april-tools/squared-npcs
    train = torch.load('./data/GPT2/train.pt')
    valid = torch.load('./data/GPT2/valid.pt')
    test = torch.load('./data/GPT2/test.pt')
    return train, valid, test


def load_mnist_family_dataset(
    ds_name: Optional[str] = 'mnist',
    root: Optional[str] = './data/',
    split: Optional[str] = None,
    transform: Optional[Callable] = None,
    unsupervised: Optional[bool] = True,
    valid_split_percentage: Optional[float] = 0
):
    assert ds_name in MNIST_DATASETS
    if transform is None:
        transform = transforms.Compose([PIC2Tensor(), ToLong(), Flatten(0, 1), UnsqueezeDim(-2), Permute((1, 0))])
    if ds_name == 'mnist':
        train = MNIST(root=root, train=True, download=True, transform=transform)
        test = MNIST(root=root, train=False, download=True, transform=transform)
    elif ds_name == 'fashion_mnist':
        train = FashionMNIST(root=root, train=True, download=True, transform=transform)
        test = FashionMNIST(root=root, train=False, download=True, transform=transform)
    else:
        assert split in ['mnist', 'letters', 'balanced', 'byclass']
        train = EMNIST(root=root, split=split, train=True, download=True, transform=transform)
        test = EMNIST(root=root, split=split, train=False, download=True, transform=transform)
    if unsupervised:
        train, test = UnsupervisedDataset(train), UnsupervisedDataset(test)
    if valid_split_percentage == 0:
        return train, test
    else:
        len_valid = int(len(train) * valid_split_percentage)
        train, valid = torch.utils.data.random_split(train, [len(train) - len_valid, len_valid], torch.Generator().manual_seed(42))
        return train, valid, test


def load_cifar10(
    root: Optional[str] = "./data/",
    transform: Optional[Callable] = None,
    unsupervised: Optional[bool] = True,
    valid_split_percentage: Optional[float] = 0,
    ycc: Optional[str] = 'none'
):
    if transform is None:
        transform = transforms.Compose([
            PIC2Tensor(),
            ToLong(),
            Flatten(0, 1),
            Permute((1, 0)),
            {'lossless': Rgb2YccLossless(), 'lossy': Rgb2YccLossy(), 'none': Identity()}[ycc],
            Permute((1, 0))
        ])
    train = CIFAR10(root=root, train=True, download=True, transform=transform)
    test = CIFAR10(root=root, train=False, download=True, transform=transform)
    if unsupervised:
        train, test = UnsupervisedDataset(train), UnsupervisedDataset(test)
    if valid_split_percentage == 0:
        return train, test
    else:
        len_valid = int(len(train) * valid_split_percentage)
        train, valid = torch.utils.data.random_split(train, [len(train) - len_valid, len_valid], torch.Generator().manual_seed(42))
        return train, valid, test


def load_imagenet(
    transform: Optional[Callable] = None,
    valid_split_percentage: Optional[float] = 0,
    ycc: Optional[str] = 'none',
    dtype: torch.dtype = torch.uint8,
    image_size: Optional[int] = 32
):
    assert image_size == 32 or image_size == 64
    if transform is None:
        transform = transforms.Compose([
            ToLong(),
            {'lossless': Rgb2YccLossless(), 'lossy': Rgb2YccLossy(), 'none': Identity()}[ycc]
        ])
    train_tensor = torch.load(
        './data/imagenet%d/train.pt' % image_size).view(-1, 3, image_size ** 2).permute(0, 2, 1).to(dtype=dtype).contiguous()
    valid_tensor = torch.load(
        './data/imagenet%d/valid.pt' % image_size).view(-1, 3, image_size ** 2).permute(0, 2, 1).to(dtype=dtype).contiguous()
    train = TensorDataset(tensor=train_tensor, transform=transform)
    test = TensorDataset(tensor=valid_tensor, transform=transform)
    if valid_split_percentage == 0:
        return train, test
    else:
        len_valid = int(len(train) * valid_split_percentage)
        train, valid = torch.utils.data.random_split(train, [len(train) - len_valid, len_valid], torch.Generator().manual_seed(42))
        return train, valid, test


class CelebADataset(Dataset):
    def __init__(self, root, split='all', ycc: Optional[str] = 'none'):
        transform = transforms.Compose([
            transforms.CenterCrop((140, 140)),
            transforms.Resize((64, 64)),
            PIC2Tensor(),
            ToLong(),
            {'lossless': Rgb2YccLossless(), 'lossy': Rgb2YccLossy(), 'none': Identity()}[ycc],
            Flatten(0, 1)
        ])
        self.dataset = CelebA(root=root, split=split, transform=transform, download=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image
