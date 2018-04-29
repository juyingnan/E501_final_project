'''Trains a simple xnor CNN on the MNIST dataset.
Modified from keras' examples/mnist_mlp.py
Gets to 98.18% test accuracy after 20 epochs using tensorflow backend
'''

from __future__ import print_function
import numpy as np
from skimage import io
import os
import random
import matplotlib.pylab as plt

np.random.seed(1337)  # for reproducibility

from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D, Conv2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.optimizers import SGD, Adam, RMSprop,Adadelta
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras import backend as K
from tensorflow import keras

K.set_image_data_format('channels_last')
# print(K.floatx())
# K.set_floatx('float16')
# print(K.floatx())

from xnornet.binary_ops import binary_tanh as binary_tanh_op
from xnornet.xnor_layers import XnorDense, XnorConv2D

# train
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


w = 100
h = 100
c = 3


def read_img_random(path, total_count, size_filter=1500):
    cate = [path + folder for folder in os.listdir(path) if os.path.isdir(path + folder)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        print('reading the images:%s' % folder)
        count = 0
        file_path_list = [os.path.join(folder, file_name) for file_name in os.listdir(folder)
                          if os.path.isfile(os.path.join(folder, file_name))]
        while count < total_count:
            im = random.choice(file_path_list)
            file_info = os.stat(im)
            file_size = file_info.st_size
            if file_size < size_filter:
                continue
            if file_size > 100 * size_filter:
                continue
            img = io.imread(im)
            if img.shape != (w, h, 3):
                continue
            imgs.append(img)
            labels.append(idx)
            count += 1
            print("\rreading {0}/{1}".format(count, total_count), end='')
        print('\r', end='')
    # return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


H = 1.
kernel_lr_multiplier = 'Glorot'

# nn
batch_size = 100
n_epoch = 200
nb_channel = 3
img_rows = 100
img_cols = 100
nb_filters = 32
nb_conv = 3
nb_pool = 2
nb_hid = 128
nb_classes = 3
train_count = 10000
test_count = 1000
use_bias = False

# learning rate schedule
lr_start = 0.0001

# BN
epsilon = 1e-7
momentum = 0.9

# dropout
p1 = 0.25
p2 = 0.5

# the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
image_path = 'c:/Users/bunny/Desktop/dataset1/root/'
(X_train, y_train) = read_img_random(path=image_path, total_count=train_count, size_filter=2000)
(X_test, y_test) = read_img_random(path=image_path, total_count=test_count, size_filter=2000)

X_train = X_train.reshape(train_count * nb_classes, img_rows, img_cols, nb_channel)
X_test = X_test.reshape(test_count * nb_classes, img_rows, img_cols, nb_channel)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = to_categorical(y_train, nb_classes)   * 2 - 1  # -1 or 1 for hinge loss
Y_test = to_categorical(y_test, nb_classes)   * 2 - 1

model = Sequential()
# tf.contrib.quantize.create_training_graph(quant_delay=2000000)
# tf.contrib.quantize.create_eval_graph()


# conv1
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(img_rows, img_cols, nb_channel),
                 strides=(1, 1),
                 activation='relu',
                 name='conv1'))
# model.add(XnorConv2D(32, kernel_size=(5, 5), input_shape=(img_rows, img_cols, nb_channel),
#                      H=H, kernel_lr_multiplier=kernel_lr_multiplier,
#                      padding='same', use_bias=use_bias, name='conv1'))
model.add(MaxPooling2D(pool_size=(4, 4), name='pool1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1'))
model.add(Activation('relu', name='act1'))
#
model.add(XnorConv2D(64, kernel_size=(3, 3), H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                     padding='same', use_bias=use_bias, name='conv2'))
model.add(MaxPooling2D(pool_size=(4, 4), name='pool2'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn2'))
model.add(Activation('relu', name='act2'))

# conv3
# model.add(XnorConv2D(128, kernel_size=(3, 3), H=H, kernel_lr_multiplier=kernel_lr_multiplier,
#                      padding='same', use_bias=use_bias, name='conv3'))
# model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))
# model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn3'))
# model.add(Activation('relu', name='act3'))
# # conv4
# model.add(XnorConv2D(256, kernel_size=(3, 3), H=H, kernel_lr_multiplier=kernel_lr_multiplier,
#                      padding='same', use_bias=use_bias, name='conv4'))
# model.add(MaxPooling2D(pool_size=(2, 2), name='pool4'))
# model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn4'))
# model.add(Activation('relu', name='act4'))
model.add(Flatten())
# dense1
model.add(XnorDense(64, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense5'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn5'))
model.add(Activation('relu', name='act5'))
# dense1
model.add(XnorDense(nb_classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense6'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn6'))
model.add(Activation('softmax', name='act6'))
# dense2
# model.add(XnorDense(nb_classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense7'))
# model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn7'))
# model.add(Activation('softmax', name='act7'))


opt = Adam(lr=lr_start)
model.compile(loss=keras.losses.categorical_hinge,#.squared_hinge,#.categorical_hinge,#.categorical_crossentropy,
              #loss='squared_hinge',
              optimizer=opt,
              metrics=['accuracy'])
model.summary()

history = AccuracyHistory()
history_callback = model.fit(X_train, Y_train,
                             batch_size=batch_size, epochs=n_epoch,
                             verbose=2, validation_data=(X_test, Y_test),
                             callbacks=[history])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, n_epoch + 1), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
import numpy

loss_history = [history_callback.history["loss"],
                history_callback.history["acc"],
                history_callback.history["val_loss"],
                history_callback.history["val_acc"]]
numpy_loss_history = numpy.array(loss_history)
numpy.savetxt("loss_history.txt", loss_history, delimiter=",")
