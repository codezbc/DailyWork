# encoding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#MNIST数据集相关的常数
INPUT_NODE=784  #输入层的节点数，对于MNIST数据集，这个就等于图片的像素
OUTPUT_NODE=10  #输出层的节点数，这个等于类别的数目，因为在MNIST数据中需要区分的是0~9这10个数字，所以这里输出层的节点数为10

#配置神经网络的参数
LAYER1_NODE=500 #隐藏层节点数，这个隐藏层有500个节点
BATCH_SIZE=100  #一个训练batch中的训练数据个数，数字越小时，训练过程越接近随机梯度下降，数字越大时，训练越接近梯度下降

LEARNING_RATE_BASE=0.8    #基础的学习率
LEARNING_RATE_DECAY=0.99  #学习率的衰减

REGULARIZATION_RATE=0.0001  #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS=30000        #训练轮数
MOVING_AVERAGE_DECAY=0.99   #滑动平均衰减率

def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):#一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果
    if avg_class==None: 
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2
    else:
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

#训练模型的过程
def train(mnist):
    x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    
    #生成隐藏层的参数
    weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    
    #生成输出层的参数
    weights2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    #计算当前参数下神经网络前向传播的结果，这里给出的用于计算滑动平均的类为None,所以函数不会使用参数的滑动平均值
    y=inference(x,None,weights1,biases1,weights2,biases2)
    
    #定义存储训练轮数的变量，这个变量不需要计算滑动平均值，所以指定这个变量为不可训练的变量
    global_step=tf.Variable(0,trainable=False)
    
    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    
    #在所有代表神经网络参数的变量上使用滑动平均。
    variable_averages_op=variable_averages.apply(tf.trainable_variables())
    
    #计算使用了滑动平均之后的前向传播结果。
    average_y=inference(x,variable_averages,weights1,biases1,weights2,biases2)
    
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))

    cross_entropy_mean=tf.reduce_mean(cross_entropy)

    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    regularization=regularizer(weights1)+regularizer(weights2)

    loss=cross_entropy_mean+regularization

    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    with tf.control_dependencies([train_step,variable_averages_op]):
       train_op=tf.no_op(name='train')

    correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    with tf.Session() as sess:
         tf.global_variables_initializer().run()

         validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}

         test_feed={x:mnist.test.images,y_:mnist.test,lables}

         for i in range(TRAINING_STEPS):
            if i%1000 ==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("After %d training step(s),validation accuracy using average model is %g"%(i,validate_acc))

            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

         test_acc=sess.run(accuracy,feed_dict=test_feed)
         print("After %d training steps(s),test accuracy using average model is %g"%(TRAINING_STEPS,test_acc))

def main(argv=None):
    mnist=input_data.read_data_sets("./mnist/data",one_hot=True)
    train(mnist)

if __name__=='__main__':
    tf.app.run()
