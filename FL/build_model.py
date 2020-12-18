import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D 
from keras.layers import AvgPool2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.constraints import maxnorm


class Model():

    def __init__(self, loss, optimizer, classes=10):
        self.loss = loss
        self.optimizer = optimizer
        self.num_classes = classes

    def mobilenet(self, train_shape):
        """
        mobileNet V2
        """
        model = Sequential()
        # 1
        #model.add(BatchNormalization())
        model.add(Conv2D(
            32, 
            kernel_size=(3,3),
            padding='SAME', 
            strides=(2,2),
            activation=tf.nn.relu,
            input_shape=train_shape
        ))
        model.add(BatchNormalization())
        # 2
        model.add(DepthwiseConv2D(
            kernel_size=(3,3),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu       
        ))
        # 3
        model.add(Conv2D(
            64, 
            kernel_size=(1,1),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        model.add(BatchNormalization())
        # 4
        model.add(DepthwiseConv2D(
            kernel_size=(3,3),
            padding='SAME', 
            strides=(2,2),
            activation=tf.nn.relu
        ))
        # 5
        model.add(Conv2D(
            128, 
            kernel_size=(1,1),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        model.add(BatchNormalization())
        # 6
        model.add(DepthwiseConv2D(
            kernel_size=(3,3),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        # 7
        model.add(Conv2D(
            128, 
            kernel_size=(1,1),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        model.add(BatchNormalization())
        # 8
        model.add(DepthwiseConv2D(
            kernel_size=(3,3),
            padding='SAME', 
            strides=(2,2),
            activation=tf.nn.relu
        ))
        # 9
        model.add(Conv2D(
            256, 
            kernel_size=(1,1),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        model.add(BatchNormalization())
        # 10
        model.add(DepthwiseConv2D(
            kernel_size=(3,3),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        # 11
        model.add(Conv2D(
            256, 
            kernel_size=(1,1),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        model.add(BatchNormalization())
        # 12
        model.add(DepthwiseConv2D(
            kernel_size=(3,3),
            padding='SAME', 
            strides=(2,2),
            activation=tf.nn.relu
        ))
        # 13
        model.add(Conv2D(
            512, 
            kernel_size=(1,1),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        model.add(BatchNormalization())
        # 14 
        model.add(DepthwiseConv2D(
            kernel_size=(3,3),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        # 15
        model.add(Conv2D(
            512, 
            kernel_size=(1,1),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        model.add(BatchNormalization())
        # 142
        model.add(DepthwiseConv2D(
            kernel_size=(3,3),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        # 152
        model.add(Conv2D(
            512, 
            kernel_size=(1,1),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        model.add(BatchNormalization())
        # 143
        model.add(DepthwiseConv2D(
            kernel_size=(3,3),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        # 153
        model.add(Conv2D(
            512, 
            kernel_size=(1,1),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        model.add(BatchNormalization())
        # 144
        model.add(DepthwiseConv2D(
            kernel_size=(3,3),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
            
        ))
        # 154
        model.add(Conv2D(
            512, 
            kernel_size=(1,1),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        model.add(BatchNormalization())
        # 145
        model.add(DepthwiseConv2D(
            kernel_size=(3,3),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        # 155
        model.add(Conv2D(
            512, 
            kernel_size=(1,1),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        model.add(BatchNormalization())
        # 16
        model.add(DepthwiseConv2D(
            kernel_size=(3,3),
            padding='SAME', 
            strides=(2,2),
            activation=tf.nn.relu
        ))
        # 17
        model.add(Conv2D(
            1024, 
            kernel_size=(1,1),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        model.add(BatchNormalization())
        # 18
        model.add(DepthwiseConv2D(
            kernel_size=(3,3),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        # 19
        model.add(Conv2D(
            1024, 
            kernel_size=(1,1),
            padding='SAME', 
            strides=(1,1),
            activation=tf.nn.relu
        ))
        # 20
        model.add(AvgPool2D(
            pool_size=(7,7), 
            strides=(1,1)
            ))
        # 21
        model.add(Dropout(
            rate=0.001
        ))
        model.add(Flatten())
        model.add(Dense(
            self.num_classes,
            activation=tf.nn.relu
        ))
        model.add(Dense(
            self.num_classes,
            activation='softmax'
        ))

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

        print("model is ok")
        return model


    def simple_model(self, train_shape):
        model = Sequential()
        #1
        model.add(Conv2D(
            filters=32,
            kernel_size=(3,3),
            padding='same',
            activation='relu',
            input_shape=train_shape
        ))
        model.add(Dropout(0.2))
        model.add(Conv2D(
            filters=32,
            kernel_size=(3,3),
            padding='same',
            activation='relu'
        ))
        model.add(MaxPooling2D(
            pool_size=(2,2),
            padding='same'
        ))
        #2
        model.add(Conv2D(
            filters=64,
            kernel_size=(3,3),
            padding='same',
            activation='relu'
        ))
        model.add(Dropout(0.2))
        model.add(Conv2D(
            filters=64,
            kernel_size=(3,3),
            padding='same',
            activation='relu'
        ))
        model.add(MaxPooling2D(
            pool_size=(2,2),
            padding='same'
        ))
        #3
        model.add(Conv2D(
            filters=128,
            kernel_size=(3,3),
            padding='same',
            activation='relu'
        ))
        model.add(Dropout(0.2))
        model.add(Conv2D(
            filters=128,
            kernel_size=(3,3),
            padding='same',
            activation='relu'
        ))
        model.add(MaxPooling2D(
            pool_size=(2,2),
            padding='same'
        ))
        #4

        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(
            units=1024,
            activation='relu',
            kernel_constraint=maxnorm(3),
        ))
        model.add(Dropout(0.3))
        model.add(Dense(
            units=512,
            activation='relu',
            kernel_constraint=maxnorm(3),
        ))
        model.add(Dropout(0.25))
        model.add(Dense(
            units=1024,
            activation='relu',
            kernel_constraint=maxnorm(3),
        ))
        #5
        model.add(Dense(
            units=self.num_classes,
            activation='softmax'
        ))

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

        return model

    def fl_paper_model(self, train_shape):
        model = Sequential()
        
        # 1
        model.add(Conv2D(
            filters=32,
            kernel_size=(5, 5),
            padding='same',
            activation='relu',
            input_shape=train_shape,
            kernel_regularizer='l2',
        ))
        model.add(Conv2D(
            filters=32,
            kernel_size=(5, 5),
            padding='same',
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(MaxPooling2D(
            pool_size=(2,2),
            padding='same'
        ))
        model.add(Dropout(0.2))

        # 2
        model.add(Conv2D(
            filters=64,
            kernel_size=(5, 5),
            padding='same',
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(Conv2D(
            filters=64,
            kernel_size=(5, 5),
            padding='same',
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(MaxPooling2D(
            pool_size=(2,2),
            padding='same'
        ))
        model.add(Dropout(0.2))

        # 3
        model.add(Flatten())
        model.add(Dense(
            units=512,
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(Dropout(0.2))
        
        # 4
        model.add(Dense(
            units=self.num_classes,
            activation='softmax'
        ))

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

        return model
    
    def deep_model(self, train_shape):
        model = Sequential()
        
        # 1
        model.add(Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            input_shape=train_shape,
            kernel_regularizer='l2',
        ))
        model.add(Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(MaxPooling2D(
            pool_size=(2,2),
            padding='same'
        ))
        model.add(Dropout(0.2))

        # 2
        model.add(Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(MaxPooling2D(
            pool_size=(2,2),
            padding='same'
        ))
        model.add(Dropout(0.2))
        
        # 3
        model.add(Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(MaxPooling2D(
            pool_size=(2,2),
            padding='same'
        ))
        model.add(Dropout(0.2))

        # 4
        model.add(Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(MaxPooling2D(
            pool_size=(2,2),
            padding='same'
        ))
        model.add(Dropout(0.2))

        # 5
        model.add(Flatten())
        model.add(Dense(
            units=1024,
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(Dense(
            units=512,
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(Dropout(0.2))
        
        # 4
        model.add(Dense(
            units=self.num_classes,
            activation='softmax'
        ))

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

        return model


    def fl_paper_model_wo_compile(self, train_shape):
        model = Sequential()
        
        # 1
        model.add(Conv2D(
            filters=32,
            kernel_size=(5,5),
            padding='same',
            activation='relu',
            input_shape=train_shape,
            kernel_regularizer='l2',
        ))
        model.add(Conv2D(
            filters=32,
            kernel_size=(5,5),
            padding='same',
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(MaxPooling2D(
            pool_size=(2,2),
            padding='same'
        ))
        model.add(Dropout(0.2))

        # 2
        model.add(Conv2D(
            filters=64,
            kernel_size=(5,5),
            padding='same',
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(Conv2D(
            filters=64,
            kernel_size=(5,5),
            padding='same',
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(MaxPooling2D(
            pool_size=(2,2),
            padding='same'
        ))
        model.add(Dropout(0.2))
        # 3
        model.add(Flatten())
        model.add(Dense(
            units=512,
            activation='relu',
            kernel_regularizer='l2',
        ))
        model.add(Dropout(0.2))
        
        # 4
        model.add(Dense(
            units=self.num_classes,
            activation='softmax'
        ))

        return model