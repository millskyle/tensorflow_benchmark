import tensorflow as tf
import numpy as np
import time
import logging
logging.basicConfig(level=logging.DEBUG)



npdtype=np.float32
tfdtype=tf.float32
L = 256

def make_data(N, L):
    X = np.random.rand(N,L,L,1)
    Y = np.random.rand(N, 1)
    return X.astype(npdtype), Y.astype(npdtype)

def reducing(_in, scope):
    """
       A reducing convolutional layer has 64 filters of size 3x3.
       We use stride 2 to half the data size.
       We use ReLU activation
    """
    with tf.variable_scope('r_' + scope, initializer=tf.glorot_normal_initializer(dtype=tfdtype)):
        out_ = tf.layers.conv2d(_in, 64, kernel_size=3, strides=(2, 2), activation=tf.nn.relu, padding='same')
        return out_

def nonreducing(_in, scope):
    """
       A nonreducing convolutional layer has 16 filters of size 4x4.
       We use stride 1 to preserve the data size.
       We use ReLU activation.
    """
    with tf.variable_scope('nr_' + scope, initializer=tf.glorot_normal_initializer(dtype=tfdtype)):
        out_ = tf.layers.conv2d(_in, 16, kernel_size=4, strides=(1, 1), activation=tf.nn.relu, padding='same')
        return out_

def CNN(_in):
    with tf.variable_scope('neuralnet', initializer=tf.glorot_normal_initializer(dtype=tfdtype)):
        net = tf.reshape(_in, (-1, L, L, 1))
        for moduleID in range(6):
            print(net.shape)
            net = reducing(net, scope=str(moduleID))
            print(net.shape)
            net = nonreducing(net, scope=str(moduleID)+"A")
            print(net.shape)
            net = nonreducing(net, scope=str(moduleID)+"B")
            print(net.shape)
        net = tf.reshape(net, (-1, 4*4*16))
        net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
        net = tf.layers.dense(net, 1, activation=None)
        return net


with tf.variable_scope('neuralnet', initializer=tf.glorot_normal_initializer(dtype=tfdtype)):
    logging.debug("Building graph...")
#data comes in a [ batch * L * L * 1 ] tensor, and labels a [ batch * 1] tensor
    x = tf.placeholder(tfdtype, (None, L, L, 1), name='input_image')
    y = tf.placeholder(tfdtype, (None, 1))
    predicted = CNN(x)
#define the loss function
    loss = tf.reduce_mean(tf.square(y-predicted))
#create an optimizer, a training op, and an init op
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
    train_step = optimizer.minimize(loss)
    init = tf.global_variables_initializer()

logging.debug("Initializing variables...")

sess = tf.InteractiveSession()
sess.run(init)

for variable in tf.trainable_variables():
    logging.debug(variable)



stime = time.time()
BATCH_SIZE = 250
EPOCHS = 1000
logging.debug("Generating random data...")
train_data, train_labels = make_data(BATCH_SIZE, L)
_, loss_val = sess.run([train_step, loss],
                    feed_dict={
                        x: train_data,
                        y: train_labels
                    }
                )

logging.debug("Beginning training...")
throughput = 0.
stime = time.time()
for epoch in range(EPOCHS):
    _, loss_val = sess.run([train_step, loss],
                    feed_dict={
                        x: train_data,
                        y: train_labels
                    }
                )

    n = (epoch+1)*BATCH_SIZE
    totaltime = time.time() - stime
    throughput = n/totaltime
    logging.info("Throughput: {0:10.4f}".format(throughput))





