import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from keras.datasets import cifar10
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.backend import image_data_format


import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import sys

from build_model import Model
import csv

# client config
NUMOFCLIENTS = 10 # number of client(as particles)
SELECT_CLIENTS = 0.5 # c
EPOCHS = 30 # number of total iteration
CLIENT_EPOCHS = 5 # number of each client's iteration
BATCH_SIZE = 10 # Size of batches to train on
DROP_RATE = 0

# model config 
LOSS = 'categorical_crossentropy' # Loss function
NUMOFCLASSES = 10 # Number of classes
lr = 0.0025
# OPTIMIZER = SGD(lr=0.015, decay=0.01, nesterov=False)
OPTIMIZER = SGD(lr=lr, momentum=0.9, decay=lr/(EPOCHS*CLIENT_EPOCHS), nesterov=False) # lr = 0.015, 67 ~ 69%


def write_csv(method_name, list):
    file_name = '{name}_CIFAR10_randomDrop_{drop}%_output_C_{c}_LR_{lr}_CLI_{cli}_CLI_EPOCHS_{cli_epoch}_TOTAL_EPOCHS_{epochs}_BATCH_{batch}.csv'
    file_name = file_name.format(folder="origin_drop",drop=DROP_RATE, name=method_name, c=SELECT_CLIENTS, lr=lr, cli=NUMOFCLIENTS, cli_epoch=CLIENT_EPOCHS, epochs=EPOCHS, batch=BATCH_SIZE)
    f = open(file_name, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    
    for l in list:
        wr.writerow(l)
    f.close()


def load_dataset():
    # Code for experimenting with CIFAR-10 datasets.
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    
    # Code for experimenting with MNIST datasets.
    # (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    # X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    # X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    return (X_train, Y_train), (X_test, Y_test)


def init_model(train_data_shape):
    model = Model(loss=LOSS, optimizer=OPTIMIZER, classes=NUMOFCLASSES)
    fl_model = model.fl_paper_model(train_shape=train_data_shape)

    return fl_model


def client_data_config(x_train, y_train):
    client_data = [() for _ in range(NUMOFCLIENTS)] # () for _ in range(NUMOFCLIENTS)
    num_of_each_dataset = int(x_train.shape[0] / NUMOFCLIENTS)
    
    for i in range(NUMOFCLIENTS):
        split_data_index = []
        while len(split_data_index) < num_of_each_dataset:
            item = random.choice(range(x_train.shape[0]))
            if item not in split_data_index:
                split_data_index.append(item)
        
        new_x_train = np.asarray([x_train[k] for k in split_data_index])
        new_y_train = np.asarray([y_train[k] for k in split_data_index])
    
        client_data[i] = (new_x_train, new_y_train)

    return client_data


def fedAVG(server_weight):
    avg_weight = np.array(server_weight[0])
    
    if len(server_weight) > 1:
        for i in range(1, len(server_weight)):
            avg_weight += server_weight[i]
    
    avg_weight = avg_weight / len(server_weight)

    return avg_weight


def client_update(index, client, now_epoch, avg_weight):
    print("client {}/{} fitting".format(index + 1, int(NUMOFCLIENTS * SELECT_CLIENTS)))

    if now_epoch != 0:
        client.set_weights(avg_weight) 
    
    client.fit(client_data[index][0], client_data[index][1],
        epochs=CLIENT_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_split=0.2,
    )

    return client


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_dataset()

    server_model = init_model(train_data_shape=x_train.shape[1:])
    server_model.summary()

    client_data = client_data_config(x_train, y_train)
    fl_model = []
    for i in range(NUMOFCLIENTS):
        fl_model.append(init_model(train_data_shape=client_data[i][0].shape[1:]))

    avg_weight = np.zeros_like(server_model.get_weights())
    server_evaluate_acc = []

    for epoch in range(EPOCHS):  
        server_weight = []
        
        selected_num = int(max(NUMOFCLIENTS * SELECT_CLIENTS, 1))
        split_data_index = []
        while len(split_data_index) < selected_num:
            item = random.choice(range(len(fl_model)))
            if item not in split_data_index:
                split_data_index.append(item)
        split_data_index.sort()
        selected_model = [fl_model[k] for k in split_data_index]

        for index, client in enumerate(selected_model):
            recv_model = client_update(index, client, epoch, avg_weight)
            
            rand = random.randint(0,99)
            drop_communication = range(DROP_RATE)
            if rand not in drop_communication:
                server_weight.append(copy.deepcopy(recv_model.get_weights()))
        
        avg_weight = fedAVG(server_weight)

        server_model.set_weights(avg_weight)
        print("server {}/{} evaluate".format(epoch + 1, EPOCHS))
        server_evaluate_acc.append(server_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1))

    write_csv("FedAvg", server_evaluate_acc)