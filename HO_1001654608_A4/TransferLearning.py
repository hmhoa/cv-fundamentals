# Hoang Ho - 1001654608
# CSE 4310-001 Fundamentals of Computer Vision
# Assignment 4 - Convolutional Neural Networks
# Due May 11, 2022 by 11:59 PM

# references https://github.com/ajdillhoff/CSE6363/blob/main/deep_learning/pl_demo/LeNetModel.py
#            https://pytorch-lightning.readthedocs.io/en/1.4.3/common/weights_loading.html
#            https://github.com/ajdillhoff/CSE6363/blob/main/deep_learning/transfer_learning.ipynb
#            https://pytorch.org/vision/stable/models.html


# Transfer learning is an effective way to leverage features learned from another task into a new task.
# Use a pre-trained model provided by torchvision and fine-tune it on the Food101 dataset.

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

# to grab a pre-trained model
import torchvision.models as models

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

class TransferLearning(pl.LightningModule):
    def __init__(self):
        super(TransferLearning, self).__init__()

        # GoogLeNet is 22 layers deep (27 with pooling layers)
        # initialize a pretrained model
        # download weights from their model zoo
        # trained this on Imagenet, so output layer is 1000 for the 1000 classes in Imagenet
        pretrained_model = models.googlenet(pretrained=True)
        num_filters = pretrained_model.fc.in_features
        layers = list(pretrained_model.children())[:-1] # exclude last layer
        self.features = nn.Sequential(*layers)

        # use pretrained model to classify food101
        self.estimator = nn.Linear(num_filters, TARGET_CLASSES)

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
            