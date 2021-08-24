# This code is highly refered from
#    https://teddylee777.github.io/tensorflow/autoencoder
# and it recommends to read following sites.
# https://excelsior-cjh.tistory.com/187
# https://junstar92.tistory.com/113
# https://techblog-history-younghunjo1.tistory.com/130
# https://blog.keras.io/building-autoencoders-in-keras.html
# https://wikidocs.net/3413

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

def get_encoder(input_shape, compressed_shape):
    X_input = Input(shape=input_shape, name='Input')
    X = X_input
    X = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', \
                data_format="channels_last", name='Encoder1')(X)
    X = BatchNormalization(name="BN1", axis=1)(X)
    X = LeakyReLU(name='LReLU1')(X)

    X = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', \
                data_format="channels_last", name='Encoder2')(X)
    X = BatchNormalization(name="BN2", axis=1)(X)
    X = LeakyReLU(name='LReLU2')(X)

    X = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', \
                data_format="channels_last", name='Encoder3')(X)
    X = BatchNormalization(name="BN3", axis=1)(X)
    X = LeakyReLU(name='LReLU3')(X)

    X = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', \
                data_format="channels_last", name='Encoder4')(X)
    X = BatchNormalization(name="BN4", axis=1)(X)
    X = LeakyReLU(name='LReLU4')(X)

    X = Flatten(name='Flattening')(X)

    encoder_output = Dense(compressed_shape[0], activation='linear', \
                name='EncoderOut')(X)
    encoder_model = Model(inputs=X_input, outputs=encoder_output, name='Encoder')

    return encoder_model

def get_decoder(input_shape, compressed_shape):
    X_input = Input(shape=compressed_shape, name='Input')
    X = X_input

    X = Dense(units=7*7*64, activation='linear', \
                name='DeFlattening')(X)
    X = Reshape((7,7,64), name='Reshape')(X)

    X = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same', \
                data_format="channels_last", name='Decoder1')(X)
    X = BatchNormalization(name="BN1", axis=1)(X)
    X = LeakyReLU(name='LReLU1')(X)

    X = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', \
                data_format="channels_last", name='Decoder2')(X)
    X = BatchNormalization(name="BN2", axis=1)(X)
    X = LeakyReLU(name='LReLU2')(X)

    X = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', \
                data_format="channels_last", name='Decoder3')(X)
    X = BatchNormalization(name="BN3", axis=1)(X)
    X = LeakyReLU(name='LReLU3')(X)

    X = Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='same', \
                data_format="channels_last", name='Decoder4')(X)
    X = BatchNormalization(name="BN4", axis=1)(X)
    X = LeakyReLU(name='LReLU4')(X)

    decoder_output = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', \
                data_format="channels_last", name='Decoder_out', activation='tanh')(X)
    decoder_model = Model(inputs=X_input, outputs=decoder_output, name='Decoder')
    
    return decoder_model

def AutoEncoder(input_shape, compressed_shape):
        input_shape = input_shape
        compressed_shape = compressed_shape

        # Hyper-parameter
        learning_rate = 0.001
        batch_size = 32

        # Define auto-encoder
        encoder = get_encoder(input_shape=input_shape, compressed_shape=compressed_shape)
        decoder = get_decoder(input_shape=input_shape, compressed_shape=compressed_shape)

        # Connect encoder with decoder
        encoder_in = Input(shape=input_shape)
        decoder_out= decoder(encoder(encoder_in))
        
        auto_encoder = Model(inputs=encoder_in, outputs=decoder_out)
        auto_encoder.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError())
        # auto_encoder.compile(optimizer=Adam(learning_rate=learning_rate), loss=BinaryCrossentropy())
        auto_encoder.summary()
        return auto_encoder, encoder, decoder

if __name__ == "__main__":
    # User Defined Choice
    data = "FASION_MNIST"
    training = False
    # Define ML model
    input_shape=(28,28,1)
    compressed_shape=(2,)
    batch_size = 32
    auto_encoder, encoder, decoder = AutoEncoder(input_shape=input_shape, compressed_shape=compressed_shape)

    # Load MNIST Data and Normalize that
    if data == "MNIST":
        mnist = tf.keras.datasets.mnist
    elif data == "FASION_MNIST":
        mnist = tf.keras.datasets.fashion_mnist
    else:
        mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    print('Shape of x_train : ', x_train.shape)
    print('Shape of y_train : ', y_train.shape)
    x_train = x_train.reshape(-1,28,28,1)
    # Input Normalization
    x_train = x_train / 127.5 - 1.
    print('Maxi of x_train : ', x_train.max())
    print('Mini of x_train : ', x_train.min())

    checkpoint_path = '01_basic_auto_encoder_' + data + '.ckpt'
    if (training==True):
        # Train
        checkpoint = ModelCheckpoint(checkpoint_path, 
                                    save_best_only=True, 
                                    save_weights_only=True, 
                                    monitor='loss', 
                                    verbose=1)
        auto_encoder.fit(x_train, x_train, 
                        batch_size=batch_size, 
                        epochs=100, 
                        callbacks=[checkpoint], 
                        )
        auto_encoder.save_weights(checkpoint_path)
        print('Save model weights')
    else:
        auto_encoder.load_weights(checkpoint_path)
        print('Load model weights')
    
    # Verify model
    # Visualize encoded data
    xy = encoder.predict(x_train)
    print('Position of the compressed data : ',xy.shape, y_train.shape)
    plt.figure(1, figsize=(15, 12))
    plt.title('Visualize encoded data' + data)
    plt.scatter(x=xy[:, 0], y=xy[:, 1], c=y_train, cmap=plt.get_cmap('Paired'), s=3)
    plt.colorbar()
    # plt.show()
    plt.savefig('Visualize encoded data ' + data + '.png')

    # Comparison of the image re-generation performance using Auto Encoder
    plt.figure(2)
    decoded_images = auto_encoder.predict(x_train)
    plt.title('Original Images' + data)
    fig, axes = plt.subplots(3, 5)
    fig.set_size_inches(12, 6)
    for i in range(15):
        axes[i//5, i%5].imshow(x_train[i].reshape(28, 28), cmap='gray')
        axes[i//5, i%5].axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig('Original Images ' + data + '.png')

    plt.figure(3)
    fig, axes = plt.subplots(3, 5)
    plt.title('Auto Encoder Images' + data)
    fig.set_size_inches(12, 6)
    for i in range(15):
        axes[i//5, i%5].imshow(decoded_images[i].reshape(28, 28), cmap='gray')
        axes[i//5, i%5].axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig('Auto Encoder Images ' + data + '.png')

    print('end')