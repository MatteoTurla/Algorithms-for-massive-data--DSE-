# Algorithms for massive data (DSE)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MatteoTurla/Algorithms-for-massive-data--DSE-/blob/main/run_project.ipynb)

Project for the course of Algorithms for massive data (DSE)

Project structure:



dataloader
	dataset.py contrains a custom TorchDataset class definition
	datautils.py contains a class use to read and menage data

model
	FeedforwardNN.py contains a torch implementation of a simple neural network
	modelutils.py contains function use to train and test a torch model

best_model.pth contain the train model
tuning_results.json contain the grid of tuned parameters and result over the validation set
train.py runnable script to tune and train the NN
test.py runnable script to test the trained model over the holdout set
run_project.ipynb a jupyter notebook runnable over google colab