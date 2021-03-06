import tensorflow as tf

from networks import conv2d
from networks import dense_block
from networks import upsampling

'''field regressor for one-to-one mapping
data flow mechanism: 64*64*1 ---> 31*31*48 ---> 31*31*128
                             ---> 17*17*64 ---> 17*17*144 
                             ---> 32*32*72 ---> 32*32*144
                             ---> 64*64*144 ---> 64*64*1 
'''

def FR_21(x, is_training=True, keep_prob=1):
    # layer 1
    with tf.variable_scope("adv", reuse=tf.AUTO_REUSE):
        pad1 = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        y = conv2d(x, 1, 48, 7, [1, 2, 2, 1], pad1, "VALID","conv1")

    # layer 2~6
        pad2 = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        y, _ = dense_block(y, 5, 48, 16, 3, [1, 1, 1, 1], pad2, "VALID", is_training, keep_prob, "dense1")
 
    # layer 7
        y = conv2d(y, 128, 64, 3, [1, 2, 2, 1], pad2, "VALID","conv2")

    # layer 8~12
        y, _ = dense_block(y, 5, 64, 16, 3, [1, 1, 1, 1], pad2, "VALID", is_training, keep_prob, "dense2")

    # layer 13~14
        y = upsampling(y, 144, 72, 3, [1, 1, 1, 1], pad2, "VALID", 64, 64, is_training, "up1")

    # layer 15~18
        y, _ = dense_block(y, 4, 72, 18, 3, [1, 1, 1, 1], pad2, "VALID", is_training, keep_prob, "dense3")

    # layer 19~20
        y = upsampling(y, 144, 144, 3, [1, 1, 1, 1], pad2, "VALID", 64, 64, is_training, "up2")

    # layer 21
        y = conv2d(y, 144, 1, 3, [1, 1, 1, 1], pad2, "VALID","conv3")
    return y


