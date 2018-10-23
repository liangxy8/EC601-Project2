#!/usr/bin/python
import tensorflow as tf
import os
import numpy as np
from random import shuffle
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt


# check the images of cat and dog
CATEGORIES = ['Dog', 'Cat']
Num_categories = len(CATEGORIES)
Train_dir = '/home/lxy/GitHub/EC601-Project2/Build_First_Neural_network/train'
Test_dir = '/home/lxy/GitHub/EC601-Project2/Build_First_Neural_network/test'


def label_image(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat':
        return [1,0]
    elif word_label == 'dog':
        return [0,1]

def creat_train_data():
    #converts the data into array data of the image and its label.
    training_data = []
    for img in tqdm(os.listdir(Train_dir)):
        label = label_image(img)
        path = os.path.join(Train_dir, img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50,50))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    #This is the actual competition test data, NOT the data that we'll use to check the accuracy of our algorithm as we test. This data has no label
    testing_data = []
    for img in tqdm(os.listdir(Test_dir)):
        path = os.path.join(Test_dir, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(50,50))
        testing_data.append([np.array(img),img_num])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

# run the training
#train_data = creat_train_data()
train_data = np.load('train_data.npy')



# define our neural network
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
tf.reset_default_graph()

convnet = input_data(shape=[None, 50, 50, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')



# set the model name
MODEL_NAME = 'dog_cat_classification-{}-{}.model'.format(1e-3, '2_layers')
print(MODEL_NAME)
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

# split out training and testing data
train = train_data[:-500]
test = train_data[-500:]

# create our data arrays, X is the images data, Y is the label
X = np.array([i[0] for i in train]).reshape(-1,50,50,1)
Y = [i[1] for i in train]

# for testing accuarcy, with label
test_x = np.array([i[0] for i in test]).reshape(-1,50,50,1)
test_y= [i[1] for i in test]

# fit for 2 epochs
model.fit({'input':X},{'targets':Y}, n_epoch=2, validation_set=({'input':test_x},{'targets':test_y}),
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# save my model
model.save(MODEL_NAME)

# visually inspecting our network against unlabeled data
import matplotlib.pyplot as plt

if os.path.exists('test_data.npy'):
    test_data = np.load('test_data.npy')
else:
    test_data = process_test_data()

fig = plt.figure()
for num, data in enumerate(test_data[:12]):   # cat:[1,0], dog:[0,1]
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(50,50,1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label='Cat'
    
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()






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
##
## define placeholder for inputs to network
#xs = tf.placeholder(tf.float32,[None,2500]) #50*50
#ys = tf.placeholder(tf.float32,[None,2])
#
## add output layer
#prediction = add_layer(xs, 2500, 2, activation_function=tf.nn.softmax)
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
#    batch_xs, batch_ys = train.next_batch(100)
#    sess.run(train_step, feed_dict={xs:batch_xs,ys:batch_ys})
#    if i%50 == 0:
#        print(computer_accuracy(test_x, test_y))
##        print(batch_xs, batch_ys)
#
