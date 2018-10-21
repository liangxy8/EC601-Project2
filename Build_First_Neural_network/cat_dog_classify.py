#!/usr/bin/python
import tensorflow as tf
import os
import numpy as np
from random import shuffle
from tqdm import tqdm

#from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# check the images of cat and dog
CATEGORIES = ['Dog', 'Cat']
Num_categories = len(CATEGORIES)
Train_dir = '/home/lxy/GitHub/EC601-Project2/Build_First_Neural_network/train.zip'
Test_dir = '/home/lxy/GitHub/EC601-Project2/Build_First_Neural_network/test.zip'


def label_image(image):
    word_label = img.split('.')[-3]
    if word_label == 'cat':
        return [1,0]
    elif word_label == 'dog':
        return [0,1]

def creat_train_data():
    #converts the data into array data of the image and its label.
    training_date = []
    for img in tqdm(os.listdir(Train_dir)):
        label = label_image(img)
        path = os.path.join(Train_dir, img)
        img = cv2.imread(path.cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50,50))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    #This is the actual competition test data, NOT the data that we'll use to check the accuracy of our algorithm as we test. This data has no label
    testing_data = []
    for img in tqdm(os.listdir(Tesr_dir)):
        #label = label_image(img)
        path = os.path.join(Test_dir, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(50,50))
        resting_data.append([np.array(img),img_num])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

# run the training
train_data = creat_train_data()




#def add_layer(inputs, in_size, out_size, activation_function=None,):
#    # add one more layer and return the output of this layer
#    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
#    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
#    Wx_plus_b = tf.matmul(inputs, Weights) + biases
#    if activation_function is None:
#        outputs = Wx_plus_b
#    else:
#        outputs = activation_function(Wx_plus_b,)
#    return outputs
#
#def computer_accuracy(v_xs, v_ys):
#    global prediction
#    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
#    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
#    return result
#
#
## define placeholder for inputs to network
#xs = tf.placeholder(tf.float32,[None,784]) #28*28
#ys = tf.placeholder(tf.float32,[None,10])
#
## add output layer
#prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
#
## the error between prediction and real data
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                             reduction_indices=[1])) # loss
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
#sess = tf.Session()
#sess.run(tf.initialize_all_variables())
#
#for i in range(1000):
#    batch_xs, batch_ys = mnist.train.next_batch(100)
#    sess.run(train_step, feed_dict={xs:batch_xs,ys:batch_ys})
#    if i%50 == 0:
#        print(computer_accuracy(mnist.test.images, mnist.test.labels))
#        print(batch_xs, batch_ys)
#
