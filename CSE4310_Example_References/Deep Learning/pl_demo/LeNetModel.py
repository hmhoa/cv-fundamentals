import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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


class LeNetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # image features that are going to be trained when we train the model
        # 3rd parameters for conv2d is the filter size (kernel?)
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5), # nn.Conv2d(1, 6, 5) -> nn.Conv2d(3, 6, 5)
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # densely connected networks
        self.estimator = nn.Sequential(
            nn.Linear(400, 120), # nn.Linear(256, 120) -> nn.Linear(400, 120)
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    # what happens if we get some raw input x, how does our model process this data?
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)

        return self.estimator(x)

    def training_step(self, batch, batch_idx):
        x, y = batch # from our batch we get our inputs and our outputs
        y_hat = self(x) # process input x
        loss = F.cross_entropy(y_hat, y) # y_hat - our model's estimate, y - ground truth

        self.log("train_loss", loss) # track model
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        prec = accuracy(y_hat, y)

        self.log("val_accuracy", prec[0])
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        prec = accuracy(y_hat, y)

        self.log("test_accuracy", prec[0])
        self.log("test_loss", loss)

    def configure_optimizers(self):
        # Adam optimizer
        # pass in parameters of the model so it knows how to update them
        # it knows which parameters to use and give it a learning rate (lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
