# Algorithms for massive data (DSE)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MatteoTurla/Algorithms-for-massive-data--DSE-/blob/main/run_project.ipynb)

Project for the course of Algorithms for massive data (DSE)

Project structure:

dataloader \
 &emsp; dataset.py contrains a custom TorchDataset class definition \
 &emsp; datautils.py contains a class use to read and menage data

model \
 &emsp; FeedforwardNN.py contains a torch implementation of a simple &emsp; neural network \
 &emsp; modelutils.py contains function use to train and test a torch model \

best_model.pth &emsp; contain the train model \
tuning_results.json &emsp; contain the results of tuning \
train.py &emsp; runnable script to tune and train the NN \
test.py &emsp; runnable script to test the trained model over the holdout set \
run_project.ipynb &emsp; a jupyter notebook runnable over google colab
