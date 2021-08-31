# Find RL_Note path and append sys path
import os, sys
cwd = os.getcwd()
dir_name = 'RL_Note'
pos = cwd.find(dir_name)
root_path = cwd[0:pos] + dir_name
sys.path.append(root_path)
print(root_path)
workspace_path = root_path + "\\pys"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, LeakyReLU, ReLU
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.activations import tanh as Tanh
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop
# from tensorflow.keras.metrics import Accuracy, MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint

from pys.utils.ER import ReplayMemory
from pys.utils.PER import ProportionalPrioritizedMemory
from pys.utils.HER import HindsightMemory
from pys.model.q_network import QNetwork
from pys.env_config import env_configs

img_size = (128,96)

def get_encoder(input_shape, compressed_shape):
    X_input = Input(shape=input_shape, name='Input')
    X = X_input
    X = Conv2D(filters=32, kernel_size=4, strides=2, padding='SAME', \
                data_format="channels_last", name='Encoder1')(X)
    X = BatchNormalization(name="BN1")(X)
    X = ReLU(name='Relu1')(X)
    X = MaxPool2D(          pool_size=2,padding='SAME', name='MaxPool1')(X)

    X = Conv2D(filters=64, kernel_size=4, strides=2, padding='SAME', \
                data_format="channels_last", name='Encoder2')(X)
    X = BatchNormalization(name="BN2")(X)
    X = ReLU(name='Relu2')(X)
    X = MaxPool2D(          pool_size=2,padding='SAME', name='MaxPool2')(X)

    X = Conv2D(filters=128, kernel_size=2, strides=1, padding='SAME', \
                data_format="channels_last", name='Encoder3')(X)
    X = ReLU(name='Relu3')(X)

    X = Flatten(name='Flattening')(X)
    encoder_output = Dense(compressed_shape[0], activation='linear', \
                name='EncoderOut')(X)
    encoder_model = Model(inputs=X_input, outputs=encoder_output, name='Encoder')

    return encoder_model

def get_decoder(input_shape, compressed_shape):
    X_input = Input(shape=compressed_shape, name='Input')
    X = X_input

    X = Dense(units=6*8*128, activation='linear', \
                name='DeFlattening')(X)
    X = Reshape((6,8,128), name='Reshape')(X)

    X = Conv2DTranspose(filters=128, kernel_size=2, strides=1, padding='SAME', \
                data_format="channels_last", name='Decoder1')(X)
    X = BatchNormalization(name="BN1")(X)
    X = ReLU(name='Relu1')(X)
    X = UpSampling2D(          size=2, interpolation='nearest',\
                data_format="channels_last", name='UpSampling1')(X)

    X = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='SAME', \
                data_format="channels_last", name='Decoder2')(X)
    X = BatchNormalization(name="BN2")(X)
    X = ReLU(name='Relu2')(X)
    X = UpSampling2D(          size=2, interpolation='nearest',\
                data_format="channels_last", name='UpSampling2')(X)

    X = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='SAME', \
                data_format="channels_last", name='Decoder3')(X)
    X = ReLU(name='Relu3')(X)

    # X = Conv2DTranspose(filters=32, kernel_size=8, strides=4, padding='SAME', \
    #             data_format="channels_last", name='Decoder4')(X)
    # X = BatchNormalization(name="BN4")(X)
    # X = LeakyReLU(name='ReLU4')(X)

    decoder_output = Conv2DTranspose(filters=1, kernel_size=4, strides=1, padding='SAME', \
                data_format="channels_last", activation='tanh', name='Decoder_out')(X)
    decoder_model = Model(inputs=X_input, outputs=decoder_output, name='Decoder')
    
    return decoder_model

def AutoEncoder(input_shape, compressed_shape):
        input_shape = input_shape
        compressed_shape = compressed_shape

        # Hyper-parameter
        learning_rate = 0.001
        batch_size = 32

        # Define auto-encoder
        print('input shape : ',input_shape)
        print('compressed_shape : ',compressed_shape)
        encoder = get_encoder(input_shape=input_shape, compressed_shape=compressed_shape)
        decoder = get_decoder(input_shape=input_shape, compressed_shape=compressed_shape)
        # encoder.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError())
        # decoder.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError())
        # encoder.summary()
        # decoder.summary()

        # Connect encoder with decoder
        encoder_in = Input(shape=input_shape)
        decoder_out= decoder(encoder(encoder_in))
        
        auto_encoder = Model(inputs=encoder_in, outputs=decoder_out)
        auto_encoder.compile(   optimizer=Adam(learning_rate=learning_rate),\
                                loss=MeanSquaredError(),\
                                metrics=[['accuracy'], ['mse']])
        # auto_encoder.compile(optimizer=Adam(learning_rate=learning_rate), loss=BinaryCrossentropy())
        auto_encoder.summary()
        return auto_encoder, encoder, decoder

def get_image(img_rgb):
    img_rgb_resize = cv2.resize(img_rgb, (img_size[0],img_size[1]), interpolation=cv2.INTER_CUBIC)
    img_k_resize = cv2.cvtColor(img_rgb_resize,cv2.COLOR_RGB2GRAY)
    # img_k_resize = img_k_resize / 255.0 # scaling 0 ~ 1
    # img_k_resize = img_k_resize / 127.5 - 1. # scaling -1 ~ 1
    state = img_k_resize
    return state

if __name__ == "__main__":
    training = False
    # Define ML model
    input_shape=(img_size[1],img_size[0],1)
    compressed_shape=(2,)
    batch_size = 64
    auto_encoder, encoder, decoder = AutoEncoder(input_shape=input_shape, compressed_shape=compressed_shape)

    # Load training data
    x_train = np.load('CartPole-v1_img.npy')
    print('Shape of x_train : ', np.shape(x_train),type(x_train))
    # Input Normalization
    x_train = x_train / 127.5 - 1.
    print('Maxi of x_train : ', x_train.max())
    print('Mini of x_train : ', x_train.min())

    # Train? or Test?
    checkpoint_path = 'CartPole-v1_img.ckpt'
    if (training==True):
        # Train
        checkpoint = ModelCheckpoint(checkpoint_path, 
                                    save_best_only=True, 
                                    save_weights_only=True, 
                                    monitor='loss', 
                                    verbose=1)
        hist = auto_encoder.fit(x_train, x_train, 
                            batch_size=batch_size, 
                            epochs=20, 
                            callbacks=[checkpoint], 
                            validation_split=0.1,
                            verbose=1
                            )
        fig = plt.figure(2)
        loss_ax = plt.subplot()
        acc_ax  = plt.twinx()
        loss_ax.plot(hist.history['loss'],'b',label='loss'); loss_ax.plot(hist.history['val_loss'],'r',label='val_loss')
        loss_ax.set_xlabel('episode'); loss_ax.set_ylabel('loss')
        acc_ax.plot(hist.history['accuracy']*100,'b--',label='accuracy');acc_ax.plot(hist.history['val_accuracy'],'r--',label='val_accuracy')
        acc_ax.set_ylabel('Accuracy [%]')
        plt.legend()
        plt.grid(); plt.title('Learning Process of Auto-Encoder')
        plt.savefig('Learning Process of Auto-Encoder.jpg')

        auto_encoder.save_weights(checkpoint_path)
        print('Save model weights')
    else:
        auto_encoder.load_weights(checkpoint_path)
        print('Load model weights')

    # Verify model
    # Visualize encoded data
    xy = encoder.predict(x_train)
    print('Position of the compressed data : ',xy.shape)
    plt.figure(1, figsize=(15, 12))
    plt.title('Visualize encoded data')
    plt.plot(xy[:, 0], xy[:, 1],'.')
    # plt.show()
    plt.savefig('Visualize encoded data.png')
    
    # Comparison of the image re-generation performance using Auto Encoder
    plt.figure(2)
    decoded_images = auto_encoder.predict(x_train)
    plt.title('Original Images')
    fig, axes = plt.subplots(3, 5)
    fig.set_size_inches(12, 6)
    for i in range(15):
        axes[i//5, i%5].imshow(x_train[i].reshape(img_size[1],img_size[0]), cmap='gray')
        axes[i//5, i%5].axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig('Original Images.png')

    plt.figure(3)
    fig, axes = plt.subplots(3, 5)
    plt.title('Auto Encoder Images')
    fig.set_size_inches(12, 6)
    for i in range(15):
        axes[i//5, i%5].imshow(decoded_images[i].reshape(img_size[1],img_size[0]), cmap='gray')
        axes[i//5, i%5].axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig('Auto Encoder Images.png')