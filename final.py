import tensornetwork as tn
import tensorflow as tf
from functools import reduce

tn.set_default_backend('tensorflow')

class MPSLayer(tf.keras.layers.Layer):

    def __init__(self, MPO, activation_function):
        super(MPSLayer, self).__init__()

        self.rank = tf.rank(MPO).numpy()
        self.shape = MPO.shape
        if (self.rank%2 != 0):
            raise Exception("Tensor network is not an MPO: must have even rank.")

        self.bias = tf.Variable(tf.zeros(shape=(tuple(MPO.shape[:self.rank//2]))), name='bias', trainable=True)
        self.MPO = MPO
        self.activation = activation_function
    def call(self, inputs):
        def f(input_vec, MPO, bias, rank):

            input_vec = tf.reshape(input_vec, tuple(MPO.shape[(rank//2):]))
            
            in_idx = [i+1 for i in range(rank//2)]
            MPO_idx = [-(i+1) for i in range(rank//2)] + in_idx
            
            MPS = tn.ncon([input_vec, MPO], [in_idx, MPO_idx])

            return MPS + bias
        result = tf.vectorized_map(
            lambda vec: f(vec, self.MPO, self.bias, self.rank), inputs)
        return self.activation(tf.reshape(result, (-1, reduce((lambda x,y : x*y), self.shape[:self.rank//2])))) #perform activation
        