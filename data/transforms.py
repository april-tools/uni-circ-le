from typing import Optional, Callable, Tuple
from torch.utils import data
from PIL import Image
import numpy as np
import torch


class TensorDataset(data.Dataset):
    def __init__(
        self,
        tensor: torch.Tensor,
        transform: Optional[Callable] = None
    ):
        self.tensor = tensor
        self.transform = transform

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, index: int):
        x = self.tensor[index]
        if self.transform:
            x = self.transform(x)
        return x


class UnsupervisedDataset(data.Dataset):
    def __init__(
        self,
        dataset: data.Dataset,
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i) -> torch.Tensor:
        x = self.dataset[i][0]
        if self.transform is not None:
            x = self.transform(x)
        return x


class PIC2Tensor(object):
    def __call__(self, img: Image):
        return torch.tensor(np.array(img, dtype=np.int8), dtype=torch.uint8)


class UnsqueezeDim(object):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, tensor: torch.Tensor):
        return tensor.unsqueeze(self.dim)


class Flatten(object):
    def __init__(self, start_dim: int, end_dim: int):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, tensor: torch.Tensor):
        return tensor.flatten(start_dim=self.start_dim, end_dim=self.end_dim)


class Permute(object):
    def __init__(self, dims: Tuple[int]):
        self.dims = dims

    def __call__(self, tensor: torch.Tensor):
        return tensor.permute(self.dims)


class ToLong(object):
    def __call__(self, tensor: torch.Tensor):
        return tensor.long()


class Identity(object):
    def __call__(self, x):
        return x


class Rgb2YccLossless(object):

    def __call__(self, rgb_images: torch.Tensor):
        assert rgb_images.size(0) == 3

        def forward_lift(x, y):
            diff = (y - x) % 256
            average = (x + (diff >> 1)) % 256
            return average, diff

        red, green, blue = rgb_images[0], rgb_images[1], rgb_images[2]
        temp, co = forward_lift(red, blue)
        y, cg = forward_lift(green, temp)
        ycc_images = torch.stack([y, co, cg], dim=0)
        return ycc_images


class Ycc2RgbLossless(object):
    def __call__(self, ycc_images):
        assert ycc_images.size(0) == 3

        def reverse_lift(average, diff):
            x = (average - (diff >> 1)) % 256
            y = (x + diff) % 256
            return x, y

        y, co, cg = ycc_images[0], ycc_images[1], ycc_images[2]
        green, temp = reverse_lift(y, cg)
        red, blue = reverse_lift(temp, co)
        rgb_images = torch.stack([red, green, blue], dim=0)
        return rgb_images


class Rgb2YccLossy(object):

    def __call__(self, rgb_images: torch.Tensor):
        assert rgb_images.size(0) == 3

        dequantized_images = (rgb_images / 127.5) - 1
        red, green, blue = dequantized_images[0], dequantized_images[1], dequantized_images[2]

        red = (red + 1) / 2
        green = (green + 1) / 2
        blue = (blue + 1) / 2

        Co = red - blue
        tmp = blue + Co / 2
        Cg = green - tmp
        Y = tmp + Cg / 2
        Y = Y * 2 - 1

        transformed_images = torch.stack((Y, Co, Cg), dim=0)
        return torch.floor(((transformed_images + 1) / 2) * 256).long().clip(0, 255)


class Ycc2Rgb2Lossy(object):

    def __call__(self, ycc_images: torch.Tensor):
        assert ycc_images.size(0) == 3

        dequantized_images = (ycc_images / 127.5) - 1
        Y, Co, Cg = dequantized_images[0], dequantized_images[1], dequantized_images[2]

        # Convert the range of Y back to [0, 1]
        Y = (Y + 1) / 2

        tmp = Y - Cg / 2
        G = Cg + tmp
        B = tmp - Co / 2
        R = B + Co

        transformed_images = torch.stack((R, G, B), dim=0)
        rgb_images = (transformed_images * 255).long().clip(0, 255).to(torch.uint8)
        return rgb_images
