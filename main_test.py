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
test_data = sio.loadmat('Test_data_plate.mat')
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

sess = tf.Session()
saver =tf.train.import_meta_graph('./cnn_model_adv.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()
input_x = graph.get_tensor_by_name("input_x:0")
output_y = graph.get_tensor_by_name("output_y:0")
is_training = graph.get_tensor_by_name("is_training:0")
lr = graph.get_tensor_by_name("lr:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

#print([n.name for n in tf.get_default_graph().as_graph_def().node])
l2_loss = graph.get_tensor_by_name("loss_l2:0")
feed_dict = {input_x: x_test, output_y: y_test, lr: learning_rate, is_training: False, keep_prob: kp}
test_loss = np.zeros(x_test.shape[0])
for i in range(x_test.shape[0]):
    feed_dict = {input_x: x_test[i,:].reshape((1,-1)), output_y: y_test[i,:].reshape((1,-1)), lr: learning_rate, is_training: False, keep_prob: kp}
    test_loss[i] = sess.run(l2_loss,feed_dict)
print(np.mean(test_loss))

