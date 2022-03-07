import tensorflow as tf
import numpy as np


def sparse_attention_graph(size, width):
    '''
    Returns an attention adjacency matrix of <size> nodes, each node connecting to <width> others
    This gives us O(<size>*<width>) nodes, which will be linear in the size of the input (<size>)
    '''
    normal_pulls = tf.random.normal((size, width//2), stddev=1.0/size)
    normal_pulls *= size * 1.3
    nums = tf.cast(normal_pulls, tf.int32)
    positions = nums % size
    adjm = np.zeros((size, size))
    for i in range(positions.shape[0]):
        for j in positions[i]:
            adjm[i,(j+i)%size] = 1
    adjm += adjm.T
    adjm = adjm.clip(max=1.0)
    return tf.constant(adjm)

def sparse_attention(Q, K, V, G, mask=None):
    '''
    Computes attention only on the nodes connected by G
    '''
    kg = tf.matmul(K, G, transpose_a=True, b_is_sparse=True)
    qkg = tf.matmul(Q, kg, b_is_sparse=True)
    dk = tf.cast(K.shape[-1], tf.float32)
    s = qkg / tf.math.sqrt(dk)
    if mask is not None:
        s -= 100000 * mask
    attn_weights = tf.nn.softmax(s, axis=-1)
    output = tf.matmul(attn_weights, V)
    return output, attn_weights

class MultiHeadSparseAttention(tf.keras.layers.Layer):

    def __init__(self, n_heads, d_model, graph_connections):
        super(MultiHeadSparseAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.g_connections = graph_connections

        assert d_model % n_heads == 0, "Error: n_heads must divide d_model"

        self.depth = d_model // n_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

        self.graph = sparse_attention_graph(d_model, graph_connections)

    def split_heads(self, x, batch_size):
        '''
        Split input vectors so so we attend on different parts of them in parallel. eg
        123456789 -> 123 456 789
        '''
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, v, q, k, mask):
        batch_size = q.shape[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        v = self.split_heads(v, batch_size)
        k = self.split_heads(k, batch_size)

        attention, weights = sparse_attention(q, k, v, self.graph, mask)

        attention = tf.transpose(attention, perm=[0,2,1,3])
        attention = tf.reshape(attention, (batch_size, -1, self.d_model))

        output = self.dense(attention)
        return output, weights