# Hoang Ho - 1001654608
# CSE 4310-001 Fundamentals of Computer Vision
# Assignment 4 - Convolutional Neural Networks
# Due May 11, 2022 by 11:59 PM

# references https://github.com/ajdillhoff/CSE6363/blob/main/deep_learning/pl_demo/LeNetModel.py
#            https://pytorch-lightning.readthedocs.io/en/1.4.3/common/weights_loading.html

# An all convolutional model - an architecture consisting of solely convolutional layers - no densely/fully connected layers or max pooling

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

TARGET_CLASSES = 101
LEARNING_RATE = 1e-3

def accuracy(output, target, topk=(1,)):
    # computes precision@k for the specified values of k
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AllCN(pl.LightningModule):
    def __init__(self):
        super(AllCN, self).__init__()

        # image features that are going to be trained when we train the model
        # conv2d - in_channels, out_channels, filter size
        # ReLU activation function for each layer - faster training times, less reliant on input normalization; makes the network non-linear (network will be able to learn more complex info and be sure the result function is not a straight line)
        # strides - amount filter moves over an image as it convolves
        # common kernel size choice is 3x3 or 5x5 so limits number of unrelated features > generalize better
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(8, 32, 3, strides=2),
            nn.ReLU(),
            nn.Conv2d(32, 128, 5, strides=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5), 
            nn.ReLU(),
            nn.Conv2d(256, TARGET_CLASSES, 3)
        )

    # how model processes the data
    # x = input
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)

        return self.estimator(x)

    def training_step(self, batch, batch_idx):
        x, y = batch # from batch, get inputs and outputs; x = images, y = labels
        y_hat = self(x) # process input x; get outputs
        loss = F.cross_entropy(y_hat, y) # y_hat - our model's estimates, y - ground truth

        self.log("train_loss", loss) # track model on tensorboard
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        prec = accuracy(y_hat, y)

        self.log("validation_accuracy", prec[0])
        self.log("validation_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        prec = accuracy(y_hat, y)

        self.log("test_accuracy", prec[0])
        self.log("test_loss", loss)

    def configure_optimizers(self):
        # Adam optimizer - appears to be popular for deep learning
        # pass in parameters of the model so it knows how to update them
        # it knows which parameters to use and give it learning rate (lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer
            