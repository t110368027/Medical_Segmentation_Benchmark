import albumentations as A

def get_train_augmentation():
    """Add paddings to make image shape divisible by 32"""
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
    ]
    return transform


def get_st_augmentation():
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
    ]
    return transform

def get_stp_augmentation():
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
        ## perturbation 
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.8),
        A.GaussianBlur(p=0.5),
        A.Cutout(p=0.5)
    ]
    return transform

def get_s3_augmentation():
    transform = [
        A.OneOf([
            A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
        ], p=0.5),
        A.RandomBrightnessContrast(p=0.8),
        A.RandomGamma(p=0.8),
        A.ShiftScaleRotate(p=0.5),
        ## perturbation 
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.8),
        A.GaussianBlur(p=0.5),
        ## perturbation 
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.5),
        A.GaussianBlur(p=0.5),
    ]
    return transform