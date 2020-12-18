import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from keras.datasets import cifar10, mnist
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras.backend import image_data_format
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import copy

from build_model import Model
import csv
import random
import time

# client config
NUMOFCLIENTS = 10 # number of client(as particles)
EPOCHS = 30 # number of total iteration
CLIENT_EPOCHS = 5 # number of each client's iteration
BATCH_SIZE = 10 # Size of batches to train on
ACC = 0.3 # 0.4
LOCAL_ACC = 0.7 # 0.6
GLOBAL_ACC = 1.4 # 1.0
DROP_RATE = 10


# model config 
# LOSS = 'sparse_categorical_crossentropy' # Loss function
LOSS = 'categorical_crossentropy' # Loss function
NUMOFCLASSES = 10 # Number of classes
lr = 0.0025
# OPTIMIZER = SGD(lr=0.15, decay=0.99)
OPTIMIZER = SGD(lr=lr, momentum=0.9, decay=lr/(EPOCHS*CLIENT_EPOCHS), nesterov=False) # lr = 0.015, 67 ~ 69%
# OPTIMIZER = SGD(lr=lr, momentum=0.9, decay=lr/EPOCHS, nesterov=False) # 67 ~ 69%
# OPTIMIZER = Adam(learning_rate=lr)


def write_csv(method_name, list):
    file_name = 'randomDrop_{drop}%_output_{name}_LR_{lr}_CLI_{cli}_CLI_EPOCHS_{cli_epoch}_TOTAL_EPOCHS_{epochs}_BATCH_{batch}.csv'
    file_name = file_name.format(drop=DROP_RATE, name=method_name, lr=lr, cli=NUMOFCLIENTS, cli_epoch=CLIENT_EPOCHS, epochs=EPOCHS, batch=BATCH_SIZE)
    f = open(file_name, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    
    for l in list:
        wr.writerow(l)
    f.close()


def load_dataset():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

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
    init_model = model.fl_paper_model(train_shape=train_data_shape)
    # init_model = model.deep_model(train_shape=train_data_shape)
    # init_model = model.mobilenet(train_shape=train_data_shape)

    return init_model


# def client_data_config(x_train, y_train):
#     '''iid'''
#     client_data = [() for _ in range(NUMOFCLIENTS)]
#     num_of_each_dataset = int(x_train.shape[0] / NUMOFCLIENTS)
    
#     for i in range(NUMOFCLIENTS):
#         #new_x_train = x_train[i*num_of_each_dataset:(i+1)*num_of_each_dataset]
#         #new_y_train = y_train[i*num_of_each_dataset:(i+1)*num_of_each_dataset]
#         #print(i, new_x_train.shape)
#         client_data[i] = (x_train[i*num_of_each_dataset:(i+1)*num_of_each_dataset], y_train[i*num_of_each_dataset:(i+1)*num_of_each_dataset])

#     return client_data
def client_data_config(x_train, y_train):
    '''non-iid'''
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

class particle():
    def __init__(self, particle_num, client, x_train, y_train):
        # for check particle id
        self.particle_id = particle_num
        
        # particle model init
        self.particle_model = client
        
        # best model init
        self.local_best_model = client
        self.global_best_model = client

        # best score init
        self.local_best_score = 0.0
        self.global_best_score = 0.0

        self.x = x_train
        self.y = y_train

        # acc = acceleration
        self.parm = {'acc':ACC, 'local_acc':LOCAL_ACC, 'global_acc':GLOBAL_ACC}
        
        # velocities init
        self.velocities = [None] * len(client.get_weights())
        for i, layer in enumerate(client.get_weights()):
            self.velocities[i] = np.random.rand(*layer.shape) / 5 - 0.10

    def train_particle(self):
        print("particle {}/{} fitting".format(self.particle_id+1, NUMOFCLIENTS))

        # set each epoch's weight
        step_model = self.particle_model
        # step_model = self.global_best_model
        step_weight = step_model.get_weights()
        
        # new_velocities = [None] * len(step_weight)
        new_weight = [None] * len(step_weight)
        local_rand, global_rand = random.random(), random.random()

        for index, layer in enumerate(step_weight):
            new_v = self.parm['acc'] * self.velocities[index]
            new_v = new_v + self.parm['local_acc'] * local_rand * (self.local_best_model.get_weights()[index] - layer)
            new_v = new_v + self.parm['global_acc'] * global_rand * (self.global_best_model.get_weights()[index] - layer)
            self.velocities[index] = new_v
            new_weight[index] = step_weight[index] + self.velocities[index]

        step_model.set_weights(new_weight)
        
        save_model_path = 'checkpoint/checkpoint_particle_{}'.format(self.particle_id)
        mc = ModelCheckpoint(filepath=save_model_path, 
                            monitor='val_loss', 
                            mode='min',
                            save_best_only=True,
                            save_weights_only=True,
                            )
        hist = step_model.fit(x=self.x, y=self.y,
                epochs=CLIENT_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=1,
                validation_split=0.2,
                callbacks=[mc],
                )
        
        # train_score_acc = hist.history['accuracy'][-1]
        train_score_loss = hist.history['val_loss'][-1]

        step_model.load_weights(save_model_path)
        self.particle_model = step_model

        # if self.global_best_score <= train_score_acc:
        if self.global_best_score >= train_score_loss:
            self.local_best_model = step_model
            
        # return step_model, train_score_acc
        # return step_model, train_score_loss
        return self.particle_id, train_score_loss
    
    def update_global_model(self, global_best_model, global_best_score):
        if self.local_best_score < global_best_score:    
            self.global_best_model = global_best_model
            self.global_best_score = global_best_score

    def resp_best_model(self, gid):
        if self.particle_id == gid:
            return self.particle_model


def get_best_score_by_loss(step_result):
    # step_result = [[step_model, train_socre_acc],...]
    temp_score = 100000
    temp_index = 0

    for index, result in enumerate(step_result):
        if temp_score > result[1]:
            temp_score = result[1]
            temp_index = index

    return step_result[temp_index][0], step_result[temp_index][1]

def get_best_score_by_acc(step_result):
    # step_result = [[step_model, train_socre_acc],...]
    temp_score = 0
    temp_index = 0

    for index, result in enumerate(step_result):
        if temp_score < result[1]:
            temp_score = result[1]
            temp_index = index

    return step_result[temp_index][0], step_result[temp_index][1]


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_dataset()

    server_model = init_model(train_data_shape=x_train.shape[1:])
    print(server_model.summary())

    client_data = client_data_config(x_train, y_train)
    pso_model = []
    for i in range(NUMOFCLIENTS):
        # pso_model.append(particle(client=init_model(train_data_shape=x_train.shape[1:]), x_train=x_train, y_train=y_train))
        pso_model.append(particle(particle_num=i, client=init_model(train_data_shape=x_train.shape[1:]), x_train=client_data[i][0], y_train=client_data[i][1]))

    # avg_weight = np.zeros_like(server_model.get_weights())
    server_evaluate_acc = []
    global_best_model = None
    global_best_score = 0.0

    for epoch in range(EPOCHS):
        server_result = []
        start = time.time()

        for client in pso_model:
            if epoch != 0:
                # client.update_global_model(global_best_model, global_best_score)
                client.update_global_model(server_model, global_best_score)
            
            # local_model, train_score = client.train_particle()
            # server_result.append([local_model, train_score])
            pid, train_score = client.train_particle()
            rand = random.randint(0,99)
            drop_communication = range(DROP_RATE)
            if rand not in drop_communication:
                server_result.append([pid, train_score])
        
        # best score 비교 후 최적의 모델 재전송
        # global_best_model, global_best_score = get_best_score_by_loss(server_result)
        gid, global_best_score = get_best_score_by_loss(server_result)
        for client in pso_model:
            if client.resp_best_model(gid) != None:
                global_best_model = client.resp_best_model(gid)

        server_model = global_best_model

        # print("server {} / {} fitting".format(epoch+1, EPOCHS))
        # server_model.fit(x=x_train, y=y_train,
        #         epochs=CLIENT_EPOCHS,
        #         batch_size=BATCH_SIZE,
        #         verbose=1,
        #         validation_split=0.1,
        #         )
        
        print("server {}/{}  evaluate".format(epoch+1, EPOCHS))
        server_evaluate_acc.append(server_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1))

    write_csv("PSO_FL", server_evaluate_acc)