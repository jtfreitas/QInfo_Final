import time
from scipy.special import binom
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal as mv_norm
import tensornetwork as tn
from tensornetwork.tn_keras.layers import Conv2DMPO, DenseMPO, DenseDecomp
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers, regularizers, optimizers, Input
from tensorflow.keras.layers import Dense,MaxPooling2D,Conv2D,Dropout,Flatten, AveragePooling2D
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os
import json

import matplotlib.pyplot as plt
from matplotlib import cm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tn.set_default_backend('tensorflow')

class TN_Toy(tf.keras.layers.Layer):
    def __init__(self, d, bond_dim):
        super().__init__()
        self.mps1 = tf.Variable(tf.random.normal((d,bond_dim), stddev = 1/d/bond_dim), trainable=True, name='mps1')
        self.mps2 = tf.Variable(tf.random.normal((d,bond_dim, 2), stddev=1/2/bond_dim/d), trainable=True, name='mps2')
        self.bias = tf.Variable(tf.random.normal((2,), stddev = 1/2), trainable=True, name='bias')
        self.bond_dim = bond_dim
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'mps1': self.mps1.numpy(),
                    'mps2': self.mps2.numpy(),
                    'bias' : self.bias.numpy(),
                    'bon_dim' : self.bond_dim,
                    'name' : self.name
                    })
        return config

    def call(self, inputs):
        def f(in_tensor, mps1, mps2, bias):
            """
            Performs index contraction on the nodes of the MPS and the input tensor, connecting them as:

                        |
           MPS1 ----- MPS2 
             \         /
              ----x----
            The free index at MPS2 allows application of a softmax activation.
            """
            x = tn.ncon([in_tensor, mps1, mps2], [[1,2], [1,3], [2,3,-1]])
            return x + bias
        result = tf.vectorized_map(lambda vec: f(vec, self.mps1, self.mps2, self.bias), inputs)
        return tf.nn.softmax(result)

class MNIST_TN(tf.keras.layers.Layer):

    def __init__(self, bond_dim, **kwargs):
        super().__init__(**kwargs)
    # Create the variables for the layer.
        self.a_var = tf.Variable(tf.random.normal(shape=(16,bond_dim),
                                                stddev=1.0/32.0),
                                name="a", trainable=True)
        self.b_var = tf.Variable(tf.random.normal(shape=(10,63,bond_dim),
                                                stddev=1.0/32.0),
                                name="b", trainable=True)
        self.bias = tf.Variable(tf.zeros(shape=(10,)),
                            name="bias", trainable=True)
        self.bond_dim = bond_dim
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({'a_var': self.a_var.numpy(),
                    'b_var': self.b_var.numpy(),
                    'bias' : self.bias.numpy(),
                    'bon_dim' : self.bond_dim,
                    'name' : self.name
                    })
        return config

    def call(self, inputs):
    # Define the contraction.
    # We break it out so we can parallelize a batch using
    # tf.vectorized_map (see below).
        def f(input_vec, a_var, b_var, bias_var):
            # Reshape to a matrix instead of a vector.
            input_vec = tf.reshape(input_vec, (16,63))

            result = tn.ncon([input_vec, a_var, b_var], [[1,2], [1, 3], [-2, 2, 3]])
            return result + bias_var

    # To deal with a batch of items, we can use the tf.vectorized_map
    # function.
    # https://www.tensorflow.org/api_docs/python/tf/vectorized_map
        result = tf.vectorized_map(
            lambda vec: f(vec, self.a_var, self.b_var, self.bias), inputs)
        return tf.nn.softmax(tf.reshape(result, (-1, 10)))


class fruit_TN1(tf.keras.layers.Layer):

    def __init__(self, bond_dim, **kwargs):
        super().__init__(**kwargs)
          # Create the variables for the layer.
        self.a_var = tf.Variable(tf.random.normal(shape=(4, 11, bond_dim),stddev=1.0/16.0),
                    name="a",
                    trainable=True)
        self.b_var = tf.Variable(tf.random.normal(shape=(4, 256, bond_dim, bond_dim), stddev=1.0/16.0),
                    name="b",
                    trainable=True)
        self.c_var = tf.Variable(tf.random.normal(shape=(4, 11, bond_dim), stddev=1.0/16.0),
                    name="c",
                    trainable=True)
        self.bias = tf.Variable(tf.zeros(shape=(4,4,4)),
                    name="bias",
                    trainable=True)
        self.bond_dim = bond_dim
    def get_config(self):
        config = super().get_config().copy()
        config.update({'a_var': self.a_var.numpy(),
                    'b_var': self.b_var.numpy(),
                    'c_var': self.c_var.numpy(),
                    'bias' : self.bias.numpy(),
                    'bond_dim' : self.bond_dim,
                    'name' : self.name
                    })
        return config

    def call(self, inputs):
      
      
        def f(input_vec, a_var, b_var, c_var, bias_var):
            result = tn.ncon([input_vec, a_var, b_var, c_var], [[1, 2, 3], [-1, 1, 4], [-2, 3, 4, 5], [-3, 2, 5]]) 
            return result + bias_var

        result = tf.vectorized_map(
            lambda vec: f(vec, self.a_var, self.b_var, self.c_var, self.bias), inputs
        )
        return tf.nn.relu(tf.reshape(result, (-1,64)))

class fruit_TN2(tf.keras.layers.Layer):

    def __init__(self, bond_dim, **kwargs):
        super().__init__(**kwargs)
          # Create the variables for the layer.
        self.a_var = tf.Variable(tf.random.normal(shape=(4, bond_dim),stddev=1.0/16.0),
                    name="a",
                    trainable=True)
        self.b_var = tf.Variable(tf.random.normal(shape=(131, 4, bond_dim, bond_dim), stddev=1.0/16.0),
                    name="b",
                    trainable=True)
        self.c_var = tf.Variable(tf.random.normal(shape=(4, bond_dim), stddev=1.0/16.0),
                    name="c",
                    trainable=True)
        self.bias = tf.Variable(tf.zeros(shape=(131)),
                    name="bias",
                    trainable=True)
        self.bond_dim = bond_dim
    def get_config(self):
        config = super().get_config().copy()
        config.update({'a': self.a_var.numpy(),
                    'b': self.b_var.numpy(),
                    'c': self.c_var.numpy(),
                    'bias' : self.bias.numpy(),
                    'bond_dim' : self.bond_dim,
                    'name' : self.name
                    })
        return config

    def call(self, inputs):
      def f(input_vec, a_var, b_var, c_var, bias_var):
        input_vec = tf.reshape(input_vec,(4,4,4))
        result = tn.ncon([input_vec, a_var, b_var, c_var], [[1, 2, 3], [1, 4], [-1, 2, 4, 5], [3, 5]]) 
        return result + bias_var
  
          # To deal with a batch of items, we can use the tf.vectorized_map
          # function.
          # https://www.tensorflow.org/api_docs/python/tf/vectorized_map
      result = tf.vectorized_map(
        lambda vec: f(vec, self.a_var, self.b_var, self.c_var, self.bias), inputs
      )
      return tf.nn.softmax(tf.reshape(result, (-1,131)))


def feat_map(x_j, d):
    phi = np.array([np.sqrt(binom(d-1, i - 1))*np.cos(np.pi/2*x_j)**(d-i)*np.sin(np.pi/2*x_j)**(i-1) for i in range(1,d+1)])
    return phi

#First plot: the pdfs used to generate data, and the learning bound
def toy_model_plot(unit_sq, dist1, dist2, x_vals, y_est, x1, x2, x_test, labels_test):
    fig = plt.figure(figsize=(21,7), tight_layout='pad')
    ax1 = fig.add_subplot(131)
    ax1.contourf(*unit_sq, dist1, cmap=cm.Reds)
    ax1.contourf(*unit_sq, dist2, cmap=cm.Blues, alpha=0.5)
    ax1.plot(x_vals, y_est)
    ax1.set_title('Modelled PDF', fontsize=22)
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)
    ax1.grid()
    #Second plot: training data
    ax2 = fig.add_subplot(132)
    ax2.scatter(*x1.T, marker='.', color='blue')
    ax2.scatter(*x2.T, marker='x', color='orange')
    ax2.plot(x_vals, y_est)

    ax2.set_title('Training data', fontsize=22)

    #Third plot: test data
    ax3 = fig.add_subplot(133)
    ax3.scatter(*x_test[labels_test == 0].T, marker = '.', color = 'blue')
    ax3.scatter(*x_test[labels_test == 1].T, marker = 'x', color = 'orange')
    ax3.plot(x_vals, y_est)
    ax3.set_title('Test data', fontsize=22)
    fig.suptitle('Toy data generation', fontsize=26)

    #Draw the separating plane
    for ax in (ax2,ax3):
        ax.grid()
        ax.fill_between(x_vals, y_est, 0, color='orange', alpha=0.1)
        ax.fill_between(x_vals, y_est, 1, color='blue', alpha= 0.1)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
    return fig

def gen_samples(max_pts, rv_tL, rv_bR, rand_state=123):

    #Generate samples.
    #samples are filtered to be within the unit_square
    no_samples = 10000
    x1, labels1 = rv_tL.rvs(no_samples, random_state=rand_state), np.zeros(no_samples)
    x1 = x1[np.array([all([x_1 > 0, x_1 < 1, x_2 > 0, x_2 < 1]) for x_1, x_2 in x1])]
    x2, labels2 = rv_bR.rvs(no_samples, random_state=rand_state),  np.ones(no_samples)
    x2 = x2[np.array([all([x_1 > 0, x_1 < 1, x_2 > 0, x_2 < 1]) for x_1, x_2 in x2])]

    #Filter samples to a maximum of max_pts and stack them
    x1, labels1 = x1[:max_pts], labels1[:max_pts]
    x2, labels2 = x2[:max_pts], labels2[:max_pts]
    return x1, labels1, x2, labels2

def optimal_boundary(dist1, dist2, unit_sq, tolerance=1e-4):
    diff_dist = dist1 - dist2
    mask_arr = np.abs(diff_dist) < tolerance # Tolerance

    #Getting the plottable arrays of x and y.
    zero_line = np.where(mask_arr, dist1, 0)
    x_idx, y_idx = np.nonzero(zero_line)
    x_vals, y_vals = unit_sq[0,x_idx][:,0], unit_sq[1,:,y_idx][:,0]

    #perform quick quadratic fit
    def f2(x,a,b,c):
        return a*x**2 + b*x + c

    (a_opt, b_opt, c_opt), cov = curve_fit(f2, x_vals, y_vals)
    y_est = f2(x_vals, a_opt, b_opt, c_opt)
    return x_vals, y_est

def fmap(d, x_train, x_test, labels_train, labels_test):
    x_ftrain = np.array([np.outer(feat_map(x_train[i,0], d), feat_map(x_train[i,1], d)) for i, _ in enumerate(x_train)])
    x_ftest = np.array([np.outer(feat_map(x_test[i,0], d), feat_map(x_test[i,1], d)) for i, _ in enumerate(x_test)])
    #One-hot encode labels
    y_htrain, y_htest = tf.one_hot(labels_train, 2), tf.one_hot(labels_test,2)
    return x_ftrain, y_htrain, x_ftest, y_htest

def plot_loss_acc(fit_history, starting_epoch=1, fontsize=20, **kwargs):

    fig, axs = plt.subplots(2,1, **kwargs)

    for key in fit_history.keys():
        if ('Prec' in key) or ('prec' in key):
            if 'val' in key:
                axs[0].plot(range(1, len(fit_history[key][starting_epoch-1:])+1), fit_history[key][starting_epoch-1:], label='Validation set')
            else:
                axs[0].plot(range(1, len(fit_history[key][starting_epoch-1:])+1), fit_history[key][starting_epoch-1:], label='Training set')
     
        elif 'loss' in key:
            if 'val' in key:
                axs[1].plot(range(1, len(fit_history[key][starting_epoch-1:])+1), fit_history[key][starting_epoch-1:], label='_Validation set')
            else:
                axs[1].plot(range(1, len(fit_history[key][starting_epoch-1:])+1), fit_history[key][starting_epoch-1:], label='_Training set')

    axs[0].set_title("Precision",fontsize=fontsize, loc='left')
    axs[1].set_title("Loss",fontsize=fontsize, loc='left')
    for ax in axs:
        ax.grid()
        ax.set_xlim(0, len(fit_history[key][starting_epoch-1:])+1)
    fig.suptitle("Training timeline", fontsize=fontsize+4)

    fig.legend(loc='upper right', fontsize=fontsize-4)
    return fig

def build_model(d, bond_dim, opt, loss='binary_crossentropy', metrics_list = ['Precision'], show_summary=False, batch_size=16):
    inputs = tf.keras.Input(shape=(d,d), batch_size=batch_size)
    tnet = TN_Toy(d, bond_dim)(inputs)
    tnetwork = tf.keras.Model(inputs=inputs, outputs = tnet)
    if show_summary:
        tnetwork.summary()
    tnetwork.compile(optimizer=opt, loss=loss, metrics = metrics_list)
    tnetwork.bond_dim = bond_dim
    tnetwork.d = d
    return tnetwork

def decision_contour(tnetwork, x_test, labels_test, **kwargs):

    unit_sq = np.mgrid[0:1.1:0.0055, 0:1.1:0.0055]
    region = unit_sq.T.reshape(unit_sq.shape[1]*unit_sq.shape[2], 2)

    sq_feat = np.array([np.outer(feat_map(region[i,0], tnetwork.d), feat_map(region[i,1], tnetwork.d)) for i, _ in enumerate(region)])
    pred_logits = tnetwork.predict(sq_feat, batch_size=len(sq_feat), steps_per_epoch=100)
    pred_labels = np.array([np.argmax(label) for label in pred_logits])

    fig = plt.figure(**kwargs)
    ax = fig.add_axes((0,0,.9,.9))
    ax.scatter(*x_test[labels_test==1].T, cmap="Paired", label="$y_i=1$", marker='.', color=(0.9921568627450981, 0.7490196078431373, 0.43529411764705883))
    ax.scatter(*x_test[labels_test==0].T, cmap="Paired", label="$y_i=0$", marker='.', color=(0.6509803921568628, 0.807843137254902, 0.8901960784313725))
    ax.contourf(*unit_sq, pred_labels.reshape(*unit_sq.shape[1:]).T, cmap="Paired", alpha=0.3)
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.tick_params(length=0)
    fig.suptitle(f"Decision boundary on test set, $(d,m) = ({tnetwork.d}, {tnetwork.bond_dim})$", ha='center')
    return fig, ax

def decision_contours(d_list, m_list, x_train, x_test, labels_train, labels_test, n_epochs=10, **kwargs):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    if (type(d_list) == list) and (type(m_list) == list):
        fig, axs = plt.subplots(len(d_list), len(m_list), **kwargs)
        unit_sq = np.mgrid[0:1.1:0.0055, 0:1.1:0.0055]
        region = unit_sq.T.reshape(unit_sq.shape[1]*unit_sq.shape[2], 2)
        for i, d in enumerate(d_list):
            x_ftrain, y_htrain, x_ftest, y_htest = fmap(d, x_train, x_test, labels_train, labels_test)
            toy_train_ds = tf.data.Dataset.from_tensor_slices((x_ftrain, y_htrain)).shuffle(100).batch(16)
            toy_test_ds = tf.data.Dataset.from_tensor_slices((x_ftest, y_htest)).shuffle(100).batch(16)

            sq_feat = np.array([np.outer(feat_map(region[k,0], d), feat_map(region[k,1], d)) for k, _ in enumerate(region)])
            sq_feat_ds = tf.data.Dataset.from_tensor_slices(sq_feat).batch(16)
            for j, bond_dim in enumerate(m_list):
                tnetwork = build_model(d, bond_dim, 'SGD', show_summary=False, batch_size=None)
                tnetwork.fit(toy_train_ds, epochs=n_epochs, verbose=0, shuffle=True)
                #plotting history
                _, test_precision = tnetwork.evaluate(toy_test_ds, verbose=0)

                pred_logits = tnetwork.predict(sq_feat_ds)
                pred_labels = np.argmax(pred_logits, axis=1)
                sc1 = axs[i,j].scatter(*x_test[labels_test==1].T, cmap="Paired", label="$y_i=1$", marker='.', color=(0.9921568627450981, 0.7490196078431373, 0.43529411764705883))
                sc2 = axs[i,j].scatter(*x_test[labels_test==0].T, cmap="Paired", label="$y_i=0$", marker='.', color=(0.6509803921568628, 0.807843137254902, 0.8901960784313725))
                axs[i,j].contourf(*unit_sq, pred_labels.reshape(*unit_sq.shape[1:]).T, cmap="Paired", alpha=0.3)
                axs[i,j].grid(alpha=0.3)
                axs[i,j].set_xlim(0,1)
                axs[i,j].set_ylim(0,1)
                axs[i,j].set_xticks([0,1])
                axs[i,j].set_yticks([0,1])
                axs[i,j].tick_params(length=0)

                axs[i,j].set_title(f"$(d,m) = ({tnetwork.d}, {tnetwork.bond_dim})$, Score: ${test_precision*100:.2f}%$", loc='left')
        fig.suptitle('Decision boundaries', fontsize=25)
        fig.legend(handles = [sc1, sc2])
        return fig, axs
    else:
        raise TypeError("Expected lists for d and bond dimensions.")

def show_conf_matrix(conf_matrix_val, plot_title, **kwargs):
####### UPDATE THIS
    fig, axs = plt.subplots(1,1, **kwargs)
    fig.suptitle('Confusion matrix')
  
    sns.heatmap(conf_matrix_val, cmap = ('Blues'), annot = True, fmt = 'd', ax = axs, cbar=False)
    axs.set_title(plot_title)
  
    axs.tick_params(which='both', bottom=False, left = False)
    axs.set_xlabel('Predicted labels')
    axs.set_ylabel('True labels')# Confusion matrices framework 
    return fig, axs

def load_fruits_dataset(train_dir, test_dir, validation_split, batch_size, img_height, img_width):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels = 'inferred',
        label_mode='categorical',
        color_mode = 'rgb',
        validation_split= validation_split,
        subset = 'training',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels = 'inferred',
        label_mode='categorical',
        color_mode = 'rgb',
        validation_split=validation_split,
        subset = 'validation',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    print('\nTest data:')
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels = 'inferred',
        label_mode='categorical',
        color_mode = 'rgb',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_ds, val_ds, test_ds