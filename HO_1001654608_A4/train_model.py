# Hoang Ho - 1001654608
# CSE 4310-001 Fundamentals of Computer Vision
# Assignment 4 - Convolutional Neural Networks
# Due May 11, 2022 by 11:59 PM

import os
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

from Food101DataModule import Food101DataModule
from BasicCNN import BasicCNN

MAX_EPOCHS = 10

def main():
    # initilize the data module
    food_data = Food101DataModule()

    # initialize model
    model = BasicCNN()

    # Add EarlyStopping
    # automatically monitor validation loss
    # want to stop if it starts to overfit
    # patience - how many successive increases (in validation loss) to tolerate before stopping
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=5)


    # Configure Checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min"
    )

    # initialize trainer
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(model=model, datamodule=food_data)

if __name__ == "__main__":
    main()

