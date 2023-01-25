import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import albumentations as A
from augment import get_val_augmentation, get_train_augmentation
from sklearn.model_selection import train_test_split

def get_patch(imgs_list, patch_size=48, stride=6):
    """
    image_list is a numpy array (B, H, W, C)
    all get_patch operate is on tensor (B, C, H, W)
    finally return numpy array patch data set (Patch, H, W, C)
    """
    image_list = []
    imgs_list = imgs_list.transpose((0,3,1,2))
    imgs_list = torch.from_numpy(imgs_list)
    _, h, w = imgs_list[0].shape
    pad_h = stride - (h - patch_size) % stride
    pad_w = stride - (w - patch_size) % stride
    for sub1 in imgs_list:
        image = F.pad(sub1, (0, pad_w, 0, pad_h), "constant", 0)
        image = image.unfold(1, patch_size, stride).unfold(
            2, patch_size, stride).permute(1, 2, 0, 3, 4)
        image = image.contiguous().view(
            image.shape[0] * image.shape[1], image.shape[2], patch_size, patch_size)
        for sub2 in image:
            image_list.append(sub2.numpy())
    image_list = np.array(image_list)
    image_list = image_list.transpose((0,2,3,1))
    return image_list


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data_tensor, 
                 target_tensor=None, 
                 transforms=None):
        if target_tensor is not None:
            assert data_tensor.shape[:1] == target_tensor.shape[:1]
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        if transforms is not None:
            self.transforms = transforms

    def __getitem__(self, index):
        data_tensor = self.data_tensor[index]

        if self.target_tensor is None:
            if self.transforms:
                sample = self.transforms(image=data_tensor)
                data_tensor = sample['image']
            else:
                data_tensor = torch.tensor(data_tensor).permute(2, 0, 1)
            return data_tensor, data_tensor

        else:
            target_tensor = self.target_tensor[index]
            target_tensor = target_tensor.astype(float)

            if self.transforms:
                sample = self.transforms(image=data_tensor, mask=target_tensor)
                data_tensor, target_tensor = sample['image'], sample['mask']
            else:
                data_tensor = torch.tensor(data_tensor).permute(2, 0, 1)
                target_tensor = torch.tensor(target_tensor).permute(2, 0, 1)
            return data_tensor, target_tensor

    def __len__(self):
        return len(self.data_tensor)