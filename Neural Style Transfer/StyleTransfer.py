
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf


Output_dir = "Images/"
Style_image = "Images/"
Content_image = "Images/Ronaldo.jpeg"


content_image = scipy.misc.imread("Images/Ronaldo.jpeg")
imshow(content_image)

def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H*n_W,n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H*n_W,n_C]))

    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))/(4*n_H*n_W*n_C)
    return J_content

style_image = scipy.misc.imread("Images/style_image.jpeg")
imshow(style_image)

def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.transpose(tf.reshape(a_S, [n_H*n_W,n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H*n_W,n_C]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG)))/(4*n_C**2*(n_W*n_H)**2)
    return J_style_layer

STYLE_LAYERS = [
    ('conv1_1', 0.5),
    ('conv2_1', 0.3),
    ('conv3_1', 0.1),
    ('conv4_1', 0.1),
    ('conv5_1', 0.0)]

def compute_style_cost(model, STYLE_LAYERS):
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha=10, beta=40):
    J = alpha*J_content + beta*J_style
    return J


### Main code
sess = tf.InteractiveSession()

content_image = scipy.misc.imread("Images/Self.jpg")
content_image = scipy.misc.imresize(content_image, (300,400))
content_image = reshape_and_normalize_image(content_image)

style_image = scipy.misc.imread("Images/style_image.jpeg")
style_image = scipy.misc.imresize(style_image, (300,400))
style_image = reshape_and_normalize_image(style_image)

generated_image = generate_noise_image(content_image)
imshow(generated_image[0])

model = load_vgg_model("/home/santosh/TF_Workspace/pre-trained-model/imagenet-vgg-verydeep-19.mat")

sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = compute_content_cost(a_C,a_G)

sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(model, STYLE_LAYERS)
J = total_cost(J_content, J_style, alpha=10, beta=40)

optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(J)


def model_nn(sess, input_image, num_iterations = 1000):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    for i in range(num_iterations):
        _ = sess.run(train_step)
        generated_image = sess.run(model['input'])
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    # save last generated image
    save_image('output/Final_image.jpg', generated_image)

    return generated_image

model_nn(sess, generated_image)
