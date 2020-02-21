# Python STL
import os
import logging
from typing import Any, Dict, Tuple
# Image Processing
import cv2
# PyTorch
import torch
from torch.utils.data import DataLoader, Dataset
# Data augmentation
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
import numpy as np
import math
import random


class MyDataset(Dataset):
    def __init__(self,
                 data_folder: str,
                 phase: str,
                 input_shape: Tuple[int, int],
                 ):
        '''

        Parameters
        ----------
        data_folder: path to the raw dataset
        phase: 'train' or 'val'
        input_shape:   shape required to input to the net H*W
        '''

        # logger = logging.getLogger('')
        # logger.info(f"Creating {phase} dataset")

        # Root folder of the dataset
        assert os.path.isdir(data_folder)

        # logger.info(f"Datafolder: {data_folder}")
        self.root = data_folder

        # Phase of learning
        assert phase in ['train', 'val']
        self.phase = phase

        path_to_imgs = os.path.join(self.root, self.phase, "imgs")
        assert os.path.isdir(path_to_imgs)
        self.image_names_list = sorted(os.listdir(path_to_imgs))
        self.input_shape = input_shape

    def __getitem__(self, idx: int):
        image_name = self.image_names_list[idx]
        image_path = os.path.join(self.root, self.phase, "imgs", image_name)
        image = np.load(image_path).astype(float) / 255
        assert image.size != 0
        mask_name = image_name
        mask_path = os.path.join(self.root, self.phase, "masks", mask_name)
        mask = np.load(mask_path).astype(float)
        assert mask.size != 0
        image, mask = self.transform(image, mask)
        assert image.shape[0] == 3 and mask.shape[0] == 2
        assert image.shape[2] == mask.shape[2] and image.shape[1] == mask.shape[1]
        return image, mask

    def __len__(self):
        return len(self.image_names_list)

    def transform(self, image, mask):
        H, W = image.shape[0], image.shape[1]
        new_h, new_w = self.input_shape[0], self.input_shape[1]
        if H > self.input_shape[0] and W > self.input_shape[1]:
            y_min, x_min = random.randint(0, H - self.input_shape[0]), random.randint(0, W - self.input_shape[1])
            compose = Compose([transforms.Crop(x_min=x_min, y_min=y_min, x_max=x_min + self.input_shape[1],
                                               y_max=y_min + self.input_shape[0]),
                               ToTensorV2()])
            new_image = compose(image=image)['image']
            new_mask = compose(image=mask)['image']
        else:
            resize_ratio = int(math.ceil(max([new_h / float(H), new_w / float(W)])))
            h, w = H * resize_ratio, W * resize_ratio
            y_min, x_min = random.randint(0, h - self.input_shape[0]), random.randint(0, w - self.input_shape[1])
            compose = Compose([transforms.Resize(h, w),
                               transforms.Crop(x_min=x_min, y_min=y_min, x_max=x_min + self.input_shape[1],
                                               y_max=y_min + self.input_shape[0]),
                               ToTensorV2()])
            new_image = compose(image=image)['image']
            new_mask = compose(image=mask)['image']
        assert new_image.shape[0] == 3 and new_mask.shape[0] == 2
        return new_image, new_mask


def provider(data_folder: str,
             phase: str,
             batch_size: int,
             input_shape) -> DataLoader:
    """Return dataloader for the dataset

    Parameters
    ----------
    data_folder : str
        Root folder of the dataset
    phase : str
        Phase of learning
        In ['train', 'val']
    batch_size : int
        Batch size

    Returns
    -------
    dataloader: DataLoader
        DataLoader for loading data from CPU to GPU
    """
    image_dataset = MyDataset(data_folder, phase, input_shape)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
    return dataloader
