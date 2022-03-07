from sympy import Mul
from attention import MultiHeadSparseAttention

import tensorflow as tf

class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, n_heads, graph_connections, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadSparseAttention(n_heads, d_model, graph_connections)