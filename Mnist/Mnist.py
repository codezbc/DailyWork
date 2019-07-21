import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import  matplotlib.pyplot as plt
import math
import numpy as np

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

batch_size = 100
X_holder = tf.placeholder(tf.float32)  #输入图片
y_holder = tf.placeholder(tf.float32)   #真实标签

Weights = tf.Variable(tf.zeros([784, 10]))  #训练参数
biases = tf.Variable(tf.zeros([1,10]))          #训练参数
predict_y = tf.nn.softmax(tf.matmul(X_holder, Weights) + biases)  #网络预测值
loss = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(predict_y), 1)) #损失函数
optimizer = tf.train.GradientDescentOptimizer(0.5)   #梯度下降
train = optimizer.minimize(loss)   #训练目标

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

for i in range(10000):  #训练轮数
    images, labels = mnist.train.next_batch(batch_size)
    session.run(train, feed_dict={X_holder:images, y_holder:labels})
    if i % 1000 == 0:
        correct_prediction = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y_holder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_value = session.run(accuracy, feed_dict={X_holder:mnist.test.images, y_holder:mnist.test.labels})
        print('step:%d accuracy:%.4f' %(i, accuracy_value))

def drawDigit(position,image,titile,isTrue):
    plt.subplot(*position)
    plt.imshow(image.reshape(-1,28),cmap='gray_r')
    plt.axis('off')
    if not isTrue:
        plt.title(titile,color='red')
    else:
        plt.title(titile)

def  batchDraw(batchsize):
    images,labels=mnist.train.next_batch(batchsize)
    predict_labels=session.run(predict_y,feed_dict={X_holder:images,y_holder:labels})
    image_number=images.shape[0]
    row_number = math.ceil(image_number ** 0.5)
    column_number = row_number
    plt.figure(figsize=(row_number+8, column_number+8))
    for i in range(row_number):
        for j in range(column_number):
            index = i * column_number + j
            if index < image_number:
                position = (row_number, column_number, index + 1)
                image = images[index]
                actual=np.argmax(labels[index])
                predict=np.argmax(predict_labels[index])
                isTrue=actual==predict
                title = 'actual:%d\npredict:%d' % (actual,predict)
                drawDigit(position, image, title,isTrue)

batchDraw(200)
plt.show()