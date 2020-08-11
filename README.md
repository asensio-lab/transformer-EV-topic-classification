# transformer-EV-topic-classification
This repository contains the replication protocols for our porject on topic classification of electric vehicle consumer experience with deep learning.

The BERT and XLNet models have been written with the help of the fastai library built on top of PyTorch based on the protocols desribed in https://github.com/kaushaltrivedi/fast-bert. The following prerequisite tools will need to be installed :

- Python version >= 3.6
- PyTorch version 1.4.0.
- Pandas.
- Scikit-learn.
- fast-bert, fastai packages.

The training scripts for these two models are 'train_bert.py' and 'train_xlnet.py', while the scripts for testing the models are 'bert_inference.py' and 'xlnet_inference.py'. All of these scripts can be found in the folder titled 'transformers' within 'transformer-EV-topic-classification'.

The baseline deep neural network based models used in the experiments were - 1) Convolutional Neural Networks (CNNs) and 2) Recurrent Neural Networks (RNNs).

The algorithms used in this paper have been written in Python, using the frameworks Tensorflow and PyTorch. 

The CNN and LSTM models have been written in Tensorflow and will need the following prerequisite tools :
- Python version >= 3.6
- Tensorflow, tested on version 1.15.0.
- Pandas.
- Scikit-learn.
- Gensim.

The relevant scripts for running these models from the repository are 'rnn_multilabel.py' and 'cnn_multilabel.py' in the folder 'cnn_lstm' within 'transformer-EV-topic-classification'. 

# training manual for data labeling task
In addition to the code, the repo also contains a copy of the training manual that we used in the field experiment for labeling the data. It is provided through a code book which can be found at https://github.com/asensio-lab/transformer-EV-topic-classification/blob/master/training-manual/training-manual.pdf. 

Due to privacy restrictions we are not able to post the reviews dataset publicly. However, there is an open data API for getting EV charging infrastructure information (Open Charge Map).

For any other clarifications on the steps for replication, feel free to get in touch with us at sameerdharur@gatech.edu or asensio@gatech.edu.

# data for training, validation and testing
In 'anonymized_data' folder, we provide anonymized review data that are used for training, validation, and testing in this project, with their topic labels.
