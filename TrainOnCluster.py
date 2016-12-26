
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 

# Load pickled data

print("Start import software module")
import pickle
import random
import numpy as np
import tensorflow as tf
import sklearn
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import flatten
import matplotlib.image as mpimg
print("Import software module Done")


# Load pickled data
print("Start data loading")
# TODO: fill this in based on where you saved the training and testing data
# need to revise on OSU OSC
training_file = '/users/PAS0947/osu8077/new/udacity/P2-CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/train.p'
testing_file = '/users/PAS0947/osu8077/new/udacity/P2-CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
test_features, test_labels = test['features'], test['labels']

del train, test
print("Data loading done")

### To start off let's do a basic data summary.
print("Start data summary")
# TODO: number of training examples
n_train = len(X_train)

# TODO: number of testing examples
n_test = len(test_features)

# TODO: what's the shape of an image?
image_shape = X_train[0].shape

# TODO: how many classes are in the dataset
n_classes = y_train[n_train -1 ] +1

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print("Data summary done")


print("Start preprocess")
#define a min-max scaling function used to normalize the image data
def normalize_inputimage(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = 0.1
    b = 0.9
    greyscale_min = 0
    greyscale_max = 255
    return a + ( ( (image_data - greyscale_min)*(b - a) )/( greyscale_max - greyscale_min ) )

def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function was borrowed from here original by Vivek Yadav:
    https://carnd-forums.udacity.com/questions/10322627/project-2-unbalanced-data-generating-additional-data-by-jittering-the-original-image
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    
    A Random uniform distribution is used to generate different parameters for transformation
    
    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    return img

#generate new training data to make the data balance
inputs_per_class = np.bincount(y_train)
# each class will have 2800-3200 training examples 
for i in range(len(inputs_per_class)):
    add_number = 3000 + random.randint(-200, 200) - inputs_per_class[i]
    
    new_features = []
    new_labels = []
    mask = np.where(y_train == i)
    features = X_train[mask]
    for j in range(add_number):
        index = random.randint(0, inputs_per_class[i] - 1)
        new_features.append(transform_image(features[index],20, 10, 5))
        new_labels.append(i)
    X_train = np.append(X_train, new_features, axis=0)
    y_train = np.append(y_train, new_labels, axis=0)
del new_features, new_labels


#Normorlization, scale the image data from [0 255] to [0.1 0.9]
X_train = normalize_inputimage(X_train)
test_features = normalize_inputimage(test_features)

# randomly split the original training data into training and validation
train_features, validation_features, train_labels, validation_labels = train_test_split(
   X_train,
   y_train,
   test_size=0.2,
   random_state=36452
)

# One-hot encoded training and validation labels
train_labels = tf.one_hot(train_labels, n_classes).eval(session=tf.Session())
validation_labels = tf.one_hot(validation_labels, n_classes).eval(session=tf.Session())

print('preprocess done')


# Parameters
learning_rate = 0.01
batch_size = 128
training_epochs = 200
#number of classes in the German traffic sign datasets
n_classes = 43  

def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


#Prepare to train the model
# tf Graph input
x = tf.placeholder("float32", [None, 32, 32, 3])
y = tf.placeholder("float32", [None, n_classes])

logits = LeNet(x)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.AdamOptimizer().minimize(cost)


# Initializing the variable
init = tf.initialize_all_variables()
saver = tf.train.Saver()                      
number_train = len(train_labels)


# Launch the graph and train the model
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(number_train/batch_size)
        # Loop over all batches
        for i in range(total_batch):        
            #randomly seletcly batch_zise train data
            indices = np.random.random_integers(0, number_train-1, batch_size)
            batch_x, batch_y = train_features[indices], train_labels[indices]
            
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            
        # Display logs per epoch step
        c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")
    save_path = saver.save(sess, "/users/PAS0947/osu8077/new/udacity/P2-CarND-Traffic-Sign-Classifier-Project/model/model.ckpt")
    print("Model saved in file: %s" % save_path)

    # Test model, compare if predicted class equals the true class of each image
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
    print(
        "Accuracy:",
        accuracy.eval({x: validation_features, y: validation_labels}))



# launch the model being train on a cluster, and test if we save it correctly.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
saver = tf.train.Saver()
# Restore variables from disk.
saver.restore(sess, "/users/PAS0947/osu8077/new/udacity/P2-CarND-Traffic-Sign-Classifier-Project/model/model.ckpt")
print("Model restored.")
# Do some work with the model
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
print(
    "Accuracy:",
    sess.run(accuracy, feed_dict={x: validation_features, y: validation_labels}))



