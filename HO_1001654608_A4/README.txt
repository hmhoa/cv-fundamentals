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
---------------------
[TRAINING THE MODEL]
To train the model, run train_model.py

Specify which model you want to train by changing the model class initialized where the line of code is: model = [model class initialization]
	- To train the Basic CNN model, initialize using BasicCNN() in place of [model class initialization]
	- To train the All Convolutional Net, initialize using AllCN()
	- To train for transfer learning, initialize using TransferLearning()

Any regularization is added by adjusting self.transform = transforms.Compose([...]) in Food101DataModule.py
---------------------
[TESTING THE MODEL]
To test the model, run test_model.py

Similar to training the model, specify the model you want to test by changing the model name where the line of code is: model = [model name].load_from_checkpoint(checkpoint_path=args[1])
	- To test the Basic CNN model, use BasicCNN in place of [model name]
	- To test the All Convolutional Net, use AllCN
	- To test for transfer learning, use TransferLearning
