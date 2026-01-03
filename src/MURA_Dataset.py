import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np

class MURA_Dataset(Dataset):
    def __init__(self, csv_images, csv_labels, root_dir, transform=None, device="cpu"):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = pd.read_csv(csv_images, header=None)
        self.labels = pd.read_csv(csv_labels, header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.device = torch.device(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_root = os.path.join(self.root_dir,
                                self.labels.iloc[idx, 0])

        img_paths = self.images[self.images[0].str.contains(img_root)][0].to_list()
        images = [self.transform(Image.open(img_path)) for img_path in img_paths]
        sample = (pad_or_trim_list(images, 3).to(self.device), torch.tensor(self.labels.iloc[idx, 1]).to(self.device))

        return sample


def pad_or_trim_list(tensor_list, target_size):
    """
    Ensures a list of tensors is exactly `target_size` in length.

    - If the list is too short, pad with zero tensors.
    - If the list is too long, truncate extra elements.

    Args:
        tensor_list (list of torch.Tensor): List of tensors (all must have the same shape).
        target_size (int): Desired fixed size.

    Returns:
        torch.Tensor: Tensor of shape [target_size, *tensor_shape].
    """
    if len(tensor_list) == 0:
        raise ValueError("Input tensor list is empty.")

    # Determine the shape of individual tensors
    tensor_shape = tensor_list[0].shape  # Assume all tensors have the same shape

    if len(tensor_list) < target_size:
        # Pad with zero tensors
        padding = [torch.zeros(tensor_shape, dtype=tensor_list[0].dtype) for _ in range(target_size - len(tensor_list))]
        fixed_size_tensor = torch.stack(tensor_list + padding)  # Concatenate real + padded tensors
    else:
        # Truncate excess elements
        fixed_size_tensor = torch.stack(tensor_list[:target_size])

    return fixed_size_tensor
