import tensorflow as tf
import numpy as np
import  matplotlib.pyplot as plt
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

x_train=np.linspace(-1,1,100)
y_train=10*x_train+np.random.rand(x_train.shape[0])
# plt.plot(x_train,y_train,"ro",label="data")
# plt.legend()
# plt.show()
epochs=30
display_step=2
x=tf.placeholder(dtype='float',name="input")
y=tf.placeholder(dtype="float",name="label")

w=tf.Variable(initial_value=tf.random_normal([1]),name="weight")
b=tf.Variable(initial_value=tf.zeros([1]),name='bias')
z=tf.multiply(x,w)+b
tf.summary.histogram("z",z)

cost=tf.reduce_mean(tf.square(y-z))
tf.summary.scalar("cost function",cost)

optim=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
saver=tf.train.Saver(max_to_keep=2)
init=tf.global_variables_initializer()

with tf.Session()  as sess:
    sess.run(init)
    summary_merge=tf.summary.merge_all()
    f_summary=tf.summary.FileWriter(logdir="./log",graph=sess.graph)
    for epoch in range(epochs):
        for x_batch,y_batch in zip(x_train,y_train):
            sess.run(optim,feed_dict={x:x_batch,y:y_batch})
            summary_tmp = sess.run(summary_merge, feed_dict={x: x_batch, y: y_batch})  # 计算summary
            f_summary.add_summary(summary=summary_tmp, global_step=epoch)  # 写入summary
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={x: x_train, y: y_train})
            print("epoch: %d, loss: %d" % (epoch, loss))
            # 保存训练过程中的模型
            # saver.save(sess, "line_regression_model/regress.cpkt", global_step=epoch)
    print("train finished...")
    # 保存最终的模型
    saver.save(sess, "./line_regression_model/regress.cpkt")
    print("final loss:", sess.run(cost, feed_dict={x: x_train, y: y_train}))
    print("weight:", sess.run(w))
    print("bias:", sess.run(b))

    # show train data and predict data
    plt.plot(x_train, y_train, "ro", label="train")
    predict = sess.run(w) * x_train + sess.run(b)
    plt.plot(x_train, predict, "b", label="predict")
    plt.legend()
    plt.show()

