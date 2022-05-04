# Hoang Ho - 1001654608
# CSE 4310-001 Fundamentals of Computer Vision
# Assignment 4 - Convolutional Neural Networks
# Due May 11, 2022 by 11:59 PM

Packages that may be needed to be installed:
	pip install tensorflow
	pip install pytorch-lightning
	conda install -c pytorch torchvision=0.12.0

To simplify code and dataset preparation, I made a PyTorch Lightning Data Module for Food101 since I could not find a data module in the library for it: Food101DataModule.py

Under this Food101 Data Module, you can change the dataset download directory, batch size, and number of workers:
	- To change where the Food101 dataset is downloaded: Adjust constant variable PATH_FOOD_DATASET
	- To change the batch size: Adjust constant variable BATCH_SIZE
	- To change the number of workers: Adjust constant variable WORKERS

Under each model definition .py file, you can change the target classes and learning rate:
    - To change the number of target classes: Adjust constant variable TARGET_CLASSES
    - To change the learning rate: Adjust constant variable LEARNING_RATE

Under train_model.py and test_model.py, you can change the maximum epochs and number of gpus:
    - To change the maximum epochs: Adjust constant variable MAX_EPOCHS
    - To change the number of gpus to utilize: Adjust constant variable NUM_GPUS
---------------------
[TRAINING THE MODEL]
To train the model, run train_model.py like so:
    python train_model.py [MODEL_NAME]

Specify the model you want to train by replacing where it says [MODEL_NAME] with one of the following options:
    - BasicCNN
    - AllCN
    - TransferLearning

example: python train_model.py BasicCNN will train the BasicCNN model defined by BasicCNN.py

Any regularization is added by using self.transform = transforms.Compose(REGULARIZATION_TRANSFORMS) in Food101DataModule.py
No regularization is using self.transform = transforms.Compose(GENERAL_TRANSFORMS)

You can adjust these transforms under their respective constant variables at the top in Food101DataModule.py
---------------------
[TESTING THE MODEL]
To test the model, run test_model.py like so:
    python test_model.py [MODEL_NAME] [CHECKPOINT_PATH]

Similar to training the model, specify the model you want to test by replacing where it says [MODEL_NAME] with one of the following options:
    - BasicCNN
    - AllCN
    - TransferLearning

Specify the saved model or checkpoint you want to load from where it says [CHECKPOINT_PATH]

example: python test_model.py BasicCNN "./lightning_logs/version_0/checkpoints/epoch=0-step=2879.ckpt"
