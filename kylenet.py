import tensorflow as tf
import numpy as np
import time


def make_data(N, L, type_=np.float16):
    X = np.random.rand(N,L,L,1)
    Y = np.random.rand(N, 1)
    return X, Y


L = 256


def reducing(_in):
    """
       A reducing convolutional layer has 64 filters of size 3x3.
       We use stride 2 to half the data size.
       We use ReLU activation
    """
    return tf.contrib.layers.conv2d(_in, 64, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
       #             weights_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float16),
       #             biases_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float16)
                    )

def nonreducing(_in):
    """
       A nonreducing convolutional layer has 16 filters of size 4x4.
       We use stride 1 to preserve the data size.
       We use ReLU activation.
    """
    return tf.contrib.layers.conv2d(_in, 16, kernel_size=4, stride=1, activation_fn=tf.nn.relu,
        #            weights_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float16),
        #            biases_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float16)
                    )






def CNN(_in):
    net = tf.reshape(_in, (-1, L, L, 1))
    #If you're using 256x256 potentials, you'll want 6 modules.
    #We'll use 4 since we're using 64x64 potentials
    #  e.g. for 256x256 use   for moduleID in range(6):
    for moduleID in range(6):
        net = nonreducing(nonreducing(reducing(net)))
    net = tf.reshape(net, (-1, 4*4*16))
    net = tf.contrib.layers.fully_connected(net, 1024, activation_fn=tf.nn.relu,
         #           weights_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float16),
         #           biases_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float16)
                    )
    net = tf.contrib.layers.fully_connected(net, 1, activation_fn=None,
          #           weights_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float16),
          #           biases_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float16),
                 )
    return net


print "Building graph..."
#data comes in a [ batch * L * L * 1 ] tensor, and labels a [ batch * 1] tensor
x = tf.placeholder(tf.float32, (None, L, L, 1), name='input_image')
y = tf.placeholder(tf.float32, (None, 1))
predicted = CNN(x)
#define the loss function
loss = tf.reduce_mean(tf.square(y-predicted))
#create an optimizer, a training op, and an init op
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
train_step = optimizer.minimize(loss)
init = tf.global_variables_initializer()

print "Initializing variables..."

sess = tf.InteractiveSession()
sess.run(init)


stime = time.time()
BATCH_SIZE = 10
EPOCHS = 1000
print "Generating random data..."
train_data, train_labels = make_data(BATCH_SIZE, L, type_=np.float32)
print "Beginning training..."
for epoch in range(EPOCHS):
    _, loss_val = sess.run([train_step, loss],
                    feed_dict={
                        x: train_data,
                        y: train_labels
                    }
                )

    n = (epoch+1)*BATCH_SIZE
    totaltime = time.time() - stime
    print "Throughput: {0:10.4f}".format(n/totaltime)





