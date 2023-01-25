import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_val_augmentation(height,width):
    """Add paddings to make image shape divisible by 32"""
    if height and width:
        transform = [
            A.Resize(height=height, width=width, interpolation=cv2.INTER_AREA),
            ToTensorV2(transpose_mask=True),
        ]
    else:
        transform = [
            ToTensorV2(transpose_mask=True),
        ]
    return A.Compose(transform)

def get_train_augmentation(height,width):
    """Add paddings to make image shape divisible by 32"""
    if height and width:
        transform = [
            A.Resize(height=height, width=width, interpolation=cv2.INTER_AREA),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=0.5),
                # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
            ], p=0.5),
            A.RandomBrightnessContrast(p=0.8),
            A.RandomGamma(p=0.8),
            A.ShiftScaleRotate(p=0.5),
            ToTensorV2(transpose_mask=True),
        ]
    else:
        transform = [
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=0.5),
                # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
            ], p=0.5),
            A.RandomBrightnessContrast(p=0.8),
            A.RandomGamma(p=0.8),
            A.ShiftScaleRotate(p=0.5),
            ToTensorV2(transpose_mask=True),
        ]
    return A.Compose(transform)