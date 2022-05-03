# Hoang Ho - 1001654608
# CSE 4310-001 Fundamentals of Computer Vision
# Assignment 4 - Convolutional Neural Networks
# Due May 11, 2022 by 11:59 PM

# references https://pytorch-lightning-bolts.readthedocs.io/en/stable/introduction_guide.html
#            https://www.geeksforgeeks.org/understanding-pytorch-lightning-datamodules/
#            https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/datamodules.html

# group dataset preparation, splitting, and transforms into a single data module to guarentee same preprocessing, splits, and transforms

import torch
import pytorch_lightning as pl

# get Food101 Dataset and transforms
from torchvision.datasets import Food101
from torchvision import transforms

# to split training and validation data
# and data loader to create data loaders for train, validation, and test data to be returned by data module
from torch.utils.data import random_split, DataLoader

# for batch samping of dataset, wrap the dataset object in a DataLoader object

PATH_FOOD_DATASET = "./data/food/"
BATCH_SIZE = 25
WORKERS = 12 # number of CPU threads

class Food101DataModule(pl.LightningDataModule):
    # data set gives us an object that lets us sample by index and lets us set up different transformations
    # data loader class allows you to actually work with multiple threads
    # so this handles taking multiple CPU threads, fetching your data, performing preprocessing
    # workers -  how many CPU threads you have
    def __init__(self):
        super().__init__()

        self.data_dir = PATH_FOOD_DATASET
        self.batch_size = BATCH_SIZE
        self.num_workers = WORKERS

        # defining any transforms to be applied on data
        # such as data augmentations or regularization
        # normalize should be applied before ToTensor (it only works on tensors)
        # consideration to take : data set is only guarenteed their longest side of any image is going to be 512 (so it could be 512 x 194) - use PadIfNeeded
        self.transform = transforms.Compose([
            # transforms.RandAugment(), 
             transforms.Resize((224,224)), 
            # transforms.RandomCrop(224), 
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.num_classes = 101

    # download the dataset
    # point to desired dataset and ask torchvision's Food101 dataset class to download if not found
    def prepare_data(self):
        # download
        Food101(self.data_dir, download=True)

    # loads in data from file and prepares pytorch tensor datasets for each split (train, val, test)
    # expects stage arg which is used to separate logic for fit and test
    # runs across all GPUs
    # set stage to none to allow for both fit and test related setup to run - if dont mind loading in all datasets at once
    def setup(self, stage=None):
        # loading data after applying the transforms
        data = Food101(self.data_dir, transform=self.transform)

        dataset_size = len(data)
        train_size = int(dataset_size * 0.95)
        val_size = dataset_size - train_size

        # split data from training and validation
        # assign train and val datasets for use in dataloaders
        self.train_data, self.val_data = random_split(data, [train_size, val_size])

        # assign test dataset for use in dataloaders
        self.test_data = Food101(self.data_dir, split="test", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)
