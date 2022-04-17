"""Field regressor for one to one mapping
Example of Mindlin plate
"""

# Load modules
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.io as sio

from regressor import FR_21
from model import FR_model_adv

# load data
train_data = sio.loadmat('train_data_1024.mat')
x_train = train_data['material_E']
y_train = train_data['dis_all']
test_data = sio.loadmat('test_data_1000.mat')
x_test = test_data['material_E']
y_test = test_data['dis_all']


weight_decay = 7e-6
batch_size = 8
learning_rate = 0.005
epochs = 500
eps = 1e-8
kp = 1
anneal_lr_freq = 20
anneal_lr_rate = 0.75
noise_level = 0.1
adv_loss_weight = 0.2

#FR_model(x_train, y_train, x_test, y_test, weight_decay, batch_size, learning_rate, epochs, eps, kp, anneal_lr_freq, anneal_lr_rate)
FR_model_adv(x_train, y_train, x_test, y_test, weight_decay, batch_size, learning_rate, epochs, eps, kp, anneal_lr_freq, anneal_lr_rate, noise_level, adv_loss_weight)


