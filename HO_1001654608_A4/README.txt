# Hoang Ho - 1001654608
# CSE 4310-001 Fundamentals of Computer Vision
# Assignment 4 - Convolutional Neural Networks
# Due May 11, 2022 by 11:59 PM

Packages that may be needed to install:
	pip install tensorflow
	pip install pytorch-lightning
	conda install -c pytorch torchvision=0.12.0

To change where the Food101 dataset is downloaded: go to Food101DataModule.py and adjust constant variable PATH_FOOD_DATASET
To change the batch size: go to Food101DataModule.py and adjust constant variable BATCH_SIZE
To change the number of workers: go to Food101DataModule.py and adjust constant variable WORKERS

[TRAINING THE MODEL]
To train the model, run train_model.py
Specify which model you want to train by changing the model class initialized where the line of code is: model = [model class initialization]
	- To train the Basic CNN model, initialize using BasicCNN() in place of [model class initialization]
	- To train the All Convolutional Net, initialize using AllCNN()
	- To train the pre-trained model, initialize using TransferLearning()

Any regularization is added by adjusting self.transform = transforms.Compose([...]) in Food101DataModule.py

[TESTING THE MODEL]
