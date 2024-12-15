import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import tifffile as tiff
from PIL import Image
import skimage.exposure as exposure
from glob import glob
import pytorch_lightning as pl

# Input image dimensions (match this with your dataset)
max_px = 640
min_px = 401

# Define a class for data augmentation
class Transforms:
    # Training set transforms include several augmentation techniques
    def __init__(self, train=True):
        if train:
            self.tf = A.Compose([
                A.ToFloat(max_value=255),
                A.Resize(min_px, min_px),
                A.ToGray(p=1.0),
                A.Flip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5,
                                           brightness_by_max=True, p=0.7),
                A.Transpose(p=0.5),
                A.Blur(blur_limit=7, p=0.5),
                # A.MedianBlur(blur_limit=7, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, 
                                border_mode=4, value=None, mask_value=None, 
                                normalized=False, p=0.5),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, 
                                    interpolation=1, border_mode=4, value=None, 
                                    mask_value=None, p=0.5),
                A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, 
                         shear=None, interpolation=1, mask_interpolation=0, cval=0, 
                         cval_mask=0, mode=0, fit_output=False, keep_ratio=False, p=0.5),
                A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, 
                              mask_pad_val=0, fit_output=False, interpolation=1, p=0.5),
                A.Sharpen(alpha=(0.2, 0.8), lightness=(0.5, 1.0), p=0.5),
                ToTensorV2()]) # numpy HWC image -> pytorch CHW tensor 
        # Validation set transforms only include basic conversions and resizing
        else:
            self.tf = A.Compose([
                A.ToFloat(max_value=255),
                A.Resize(min_px, min_px),
                A.ToGray(p=1.0),
                ToTensorV2()])

    # Allow the class instance to be called as a function to transform images
    def __call__(self, img, *args, **kwargs):
        return self.tf(image=np.array(img))['image']

# Function to convert single-channel tif images to RGB
def tif1c_to_tif3c(path):
    """Converts single-channel tif images to RGB

    Args:
        path (string): A root folder containing original input images

    Returns:
        img_tif3c (numpy.ndarray): tif image converted to RGB
    """
    img_tif1c = tiff.imread(path)
    img_tif1c = np.array(img_tif1c)
    img_rgb = np.zeros((img_tif1c.shape[0],img_tif1c.shape[1],3),dtype=np.uint8) # blank array
    img_rgb[:,:,0] = img_tif1c # copy img 3 times to make the format of img.shape=[H, W, C]
    img_rgb[:,:,1] = img_tif1c
    img_rgb[:,:,2] = img_tif1c
    img_tif3c = img_rgb
    
    # normalize image to 8-bit range
    img_norm = exposure.rescale_intensity(img_rgb, in_range='image', out_range=(0,255)).astype(np.uint8)
    img_tif3c = img_norm
    
    print(img_tif3c.dtype)
    return img_tif3c

# Define a PyTorch Lightning data module for handling dataset
class McolonyDataModule(pl.LightningDataModule):
    def __init__(self, root: str, dl_workers: int = 0, batch_size=4, sampler: str = None):
        super().__init__()
        
        self.transforms = Transforms(train=True)
        # self.train_transforms = Transforms(train=True)
        # self.val_transforms = Transforms(train=False)
        self.root = root
        self.workers = dl_workers
        self.batch_size = batch_size
        
        # Load sampler if it exists
        if sampler is not None:
            self.sampler = torch.load(sampler)
            self.sampler.generator = None
        else:
            self.sampler = None
            
    # Setup data for training/validation/testing
    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            ds = datasets.ImageFolder(self.root, transform=self.transforms)
            # ds = datasets.ImageFolder(self.root, transform=self.train_transforms)
            # ds = datasets.ImageFolder(self.root, loader=tif1c_to_tif3c) # for imgs.tif
        if stage == "test" or stage is None:
            ds = datasets.ImageFolder(self.root, transform=self.transforms)
            # ds = datasets.ImageFolder(self.root, transform=self.val_transforms)
            
        # Create train and validation splits
        train_size = int(np.floor(len(ds)*0.7))
        val_size = int(len(ds)-int(np.floor(len(ds)*0.7)))
        self.train, self.val = random_split(ds, [train_size, val_size], torch.Generator().manual_seed(111821))

    # Define methods to retrieve data loaders for each dataset
    def train_dataloader(self):
        if self.sampler is None:
            return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=False)
        else:
            return DataLoader(self.train, batch_size=self.batch_size, sampler=self.sampler, num_workers=self.workers, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=False)

# Define a class for handling test data
class McolonyTestData(object):
        def __init__(self, root):
            self.root = root
            self.tform = A.Compose([A.ToFloat(max_value=255), A.Resize(min_px, min_px), A.ToGray(p=1.0), ToTensorV2()])
            file_list = glob(root+'*.png')
            file_list.sort()
            self.img_idx = [os.path.basename(x) for x in file_list]

        def __getitem__(self, idx):            
            ## load images and labels
            fname = self.img_idx[idx]
            im = Image.open(self.root+fname)
            im = im.convert("RGB")
            im = self.tform(image=np.array(im))
            return im, fname

        def __len__(self):
            return len(self.img_idx)
