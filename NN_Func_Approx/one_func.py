import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

weight_list = []
biases_list = []
output_list = []


def add_layer(inputs, in_size, out_size, activation_func=None):
    weights = tf.Variable(tf.random_uniform([in_size, out_size], -1.0, 1.0))
    biases = tf.Variable(tf.zeros([out_size]))
    MxtC = tf.matmul(inputs, weights) + biases
    weight_list.append(weights)
    biases_list.append(biases)
    if activation_func is None:
        outputs = MxtC
    else:
        outputs = activation_func(MxtC)

    output_list.append(outputs)
    return outputs


x_data = np.linspace(0, 10, 300)[:, np.newaxis]
noise = np.random.normal(-0.05, 0.05, x_data.shape)
#https://www.librec.net/datagen.html --is also useful to create 2D dataset
y_data = 2*np.sin(x_data) + np.cos(2*x_data - 3 ) + 3*np.log(x_data + 0.5) - 4
x_data = x_data/10
y_data = y_data/10 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 5, activation_func=tf.nn.tanh)
l2 = add_layer(l1, 5, 4, activation_func=tf.nn.tanh)
prediction = add_layer(l2, 4, 1, activation_func=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()

    for step in range(1000000):
        sess.run(train, feed_dict={xs: x_data, ys: y_data})
        if step % 1000 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            print('_______________________')
            print("epoch", step)
            outputs = sess.run(output_list, feed_dict={xs: x_data, ys: y_data})
            for i in range(0, len(weight_list)):
                print("Weight", i, '\n', sess.run(weight_list[i]))
                print("Biases", i, '\n', sess.run(biases_list[i]))
                # print("Output", i, '\n', outputs[i])

            losses = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
            print("Loss", losses)

            prediction_val = outputs[-1]
            # print(prediction_val)
            lines = ax.plot(x_data, prediction_val, 'r', lw=5)
            plt.pause(0.1)

            print("\n")
            if losses < 0.005:
                break
