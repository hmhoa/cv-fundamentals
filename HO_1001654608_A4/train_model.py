# Hoang Ho - 1001654608
# CSE 4310-001 Fundamentals of Computer Vision
# Assignment 4 - Convolutional Neural Networks
# Due May 11, 2022 by 11:59 PM

# references https://github.com/ajdillhoff/CSE6363/blob/main/deep_learning/pl_demo/train_baseline.py
#            https://github.com/hmh4608/cse4310/blob/main/CSE4310_Example_References/Deep%20Learning/transfer_learning.ipynb

import os
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

from Food101DataModule import Food101DataModule
from BasicCNN import BasicCNN

MAX_EPOCHS = 1

# initilize the data module
food_data = Food101DataModule()

# initialize model
model = BasicCNN()

# Add EarlyStopping
# automatically monitor validation loss
# want to stop if it starts to overfit
# patience - how many successive increases (in validation loss) to tolerate before stopping
early_stop_callback = EarlyStopping(monitor="validation loss",
                                    mode="min",
                                    patience=5)


# Configure Checkpoints
checkpoint_callback = ModelCheckpoint(
    monitor="validation loss",
    mode="min"
)

# initialize trainer
trainer = pl.Trainer(gpus=1, callbacks=[early_stop_callback, checkpoint_callback], max_epochs=MAX_EPOCHS)

trainer.fit(model=model, datamodule=food_data)
