import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import albumentations as A
from .augment import *
from albumentations.pytorch import ToTensorV2

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
                 transforms=None
                 ):
        
        self.data_tensor = data_tensor
        if target_tensor is not None: assert data_tensor.shape[:1] == target_tensor.shape[:1]
        self.target_tensor = target_tensor
        if transforms is not None: self.transforms = transforms

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
    
def get_h_w(data_name):
    if data_name == 'CHASEDB1':
        h, w = 960, 960
    elif data_name == 'CHUAC':
        h, w = 512, 512
    elif data_name == 'DCA1':
        h, w = 320, 320
    elif data_name == 'DRIVE':
        h, w = 576, 576
    elif data_name == 'STARE':
        h, w = 704, 704
    elif data_name == 'HRF':
        h, w = 1024, 1024
    else:
        h, w = 512, 512
    return h, w

def get_transform(rand_augment=None, stage='train', height=None, width=None):
    if stage == 'train':
        resize_tfm = [A.Resize(height=height,width=width)]
        rand_tfms = rand_augment() # returns a list of transforms
        tensor_tfms = [ToTensorV2(transpose_mask=True)]
        return A.Compose(resize_tfm + rand_tfms + tensor_tfms)
    elif stage == 'patch':
        rand_tfms = rand_augment() # returns a list of transforms
        tensor_tfms = [ToTensorV2(transpose_mask=True)]
        return A.Compose(rand_tfms + tensor_tfms)
    else:
        resize_tfm = [A.Resize(height=height,width=width)]
        tensor_tfms = [ToTensorV2(transpose_mask=True)]
        return A.Compose(resize_tfm + tensor_tfms)
    
class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, 
                 batch_size: int = 8, 
                 numworkers: int = 0, 
                 data_name=None,
                 is_patch=None,
                 transform=None,
                 ):
        super().__init__()
        self.data_dir = data_dir  # labeled image, mask
        self.batch_size = batch_size
        self.numworkers = numworkers
        self.data_name = data_name  # DRIVE, STARE, DB1, HRF, ......
        self.is_patch = is_patch
        if transform is None: get_train_augmentation
        self.transform = transform
        
        if batch_size < 5:
            self.val_batch_size = 5
        else:
            self.val_batch_size = batch_size - (batch_size % 5)
            
        h, w = get_h_w(self.data_name)
        if is_patch:
            self.train_aug = get_transform(rand_augment=self.transform, stage='patch')
        else:
            self.train_aug = get_transform(rand_augment=self.transform, stage='train', height=h, width=w)
        self.val_aug = get_transform(rand_augment=None, stage='valid', height=h, width=w)  
    
    def setup(self, stage=None):
        with np.load(self.data_dir, allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_val, y_val = f['x_val'], f['y_val']
        print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
        if self.is_patch:
            x_train, y_train = get_patch(x_train), get_patch(y_train)
            print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
        self.train_dataset = ImageDataset(x_train, y_train, self.train_aug)
        self.valid_dataset = ImageDataset(x_val, y_val, self.val_aug)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.numworkers,
                          pin_memory=True,
                          drop_last=True)
    
    def val_dataloader(self):
        #  Generating val_dataloader
        return DataLoader(self.valid_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.numworkers,
                          pin_memory=True,
                          drop_last=False)
        

class DataModule_ST(pl.LightningDataModule):
    def __init__(self, data_dir: str,
                 data_dir_u: str = None,
                 data_dir_p: str = None,
                 batch_size: int = 8, 
                 numworkers: int = 0, 
                 data_name=None,
                 is_patch=None,
                 transform=None,
                 rel=None,
                 ):
        super().__init__()
        self.data_dir = data_dir  # labeled image, mask
        self.batch_size = batch_size
        self.numworkers = numworkers
        self.data_name = data_name  # DRIVE, STARE, DB1, HRF, ......
        self.is_patch = is_patch
        if transform is None: get_train_augmentation
        self.transform = transform
        self.data_dir_u = data_dir_u
        self.data_dir_p = data_dir_p
        self.rel = rel
        if batch_size < 5:
            self.val_batch_size = 5
        else:
            self.val_batch_size = batch_size - (batch_size % 5)
            
        h, w = get_h_w(self.data_name)
        if is_patch:
            self.train_aug = get_transform(rand_augment=self.transform, stage='patch')
        else:
            self.train_aug = get_transform(rand_augment=self.transform, stage='train', height=h, width=w)
        self.val_aug = get_transform(rand_augment=None, stage='valid', height=h, width=w)  
    
    def setup(self, stage=None):
        with np.load(self.data_dir, allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_val, y_val = f['x_val'], f['y_val']
        with np.load(self.data_dir_u, allow_pickle=True) as f:
            x_u, x_u_name = f['image'], f['image_name']
        with np.load(self.data_dir_p, allow_pickle=True) as f:
            y_u, y_u_name = f['image'], f['image_name']
        print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
        if self.rel is not None:
            rel_x, rel_x_name = [], []
            for id_y in y_u_name:
                for idx, id_x in enumerate(x_u_name):
                    if id_y == id_x:
                        rel_x.append(x_u[idx])
                        rel_x_name.append(x_u_name[idx])
            del x_u, x_u_name
            x_u, x_u_name = np.array(rel_x), np.array(rel_x_name)
        print(x_u.shape, y_u.shape)
        transform = A.Compose([A.Resize(height=x_u[0].shape[0], width=x_u[0].shape[1], interpolation=cv2.INTER_AREA)])
        y_u_ = []
        for img in y_u:
            y_u_.append(transform(image=img)['image'])
        y_u = np.array(y_u_)
        if self.is_patch:
            x_train, y_train = get_patch(x_train), get_patch(y_train)
            x_u, y_u = get_patch(x_u), get_patch(y_u)
            print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
            print(x_u.shape, y_u.shape)
        x_semi, y_semi = np.concatenate((x_train, x_u)), np.concatenate((y_train, y_u))
        self.train_dataset = ImageDataset(x_semi, y_semi, self.train_aug)
        self.valid_dataset = ImageDataset(x_val, y_val, self.val_aug)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.numworkers,
                          pin_memory=True,
                          drop_last=True)
    
    def val_dataloader(self):
        #  Generating val_dataloader
        return DataLoader(self.valid_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.numworkers,
                          pin_memory=True,
                          drop_last=False)

        
if __name__ == '__main__':
    # from augment import get_train_augmentation
    # dataset = DataModule(
    #     data_dir = 'datasets/CHUAC/set.npz', 
    #     batch_size = 512,
    #     numworkers = 0,
    #     data_name = "CHUAC",
    #     is_patch = True,
    #     transform = get_train_augmentation)
    # dataset.setup()
    # train_dataloader = dataset.val_dataloader()
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    pass