from glob import glob
import tensorflow as tf
import numpy as np
import time

def unique_connections(T):
    unique, _ = tf.unique(T)
    l = T.shape[0]
    i = 0
    d = l - unique.shape[0]
    steps = 0
    while d > 0:
        steps += 1
        globs = tf.range(i, i+d)
        i = i + d
        unique = tf.concat([globs, unique], axis = 0)
        unique = tf.sort(unique)
        unique, _ = tf.unique(unique)
        d = l - unique.shape[0]
    return unique

def sparse_attention_graph(size, width, global_tokens):
    '''
    Returns an directed attention adjacency list of <size> nodes, each node connecting to <width> others
    This gives us O(<size>*<width>) nodes, which will be linear in the size of the input (<size>).
    
    The connections are made by pulling from a normal distribution. If by chance a node is adjacent to 
    the same node twice, the second (and subsequent) connections are moved to the beginning of the input, 
    creating gloabl-ish nodes there. There are expected to be <global_tokens> global-ish tokens.
    '''
    # See thesis section 4.3 for the math of this
    r = 4.0*tf.math.erfinv(2.0*global_tokens/(width*(width-1)))
    normal_pulls = tf.random.normal((size, width), stddev=1/r)
    adjlist = tf.cast(normal_pulls, tf.int32)
    adjlist += tf.reshape(tf.range(size), (size, 1))
    adjlist = adjlist % size
    adjlist = tf.sort(adjlist, axis=-1)
    adjlist = tf.map_fn(unique_connections, adjlist)
    adjlist = tf.sort(adjlist, axis=-1)
    return adjlist
    

def timeme():
    s = time.time()
    g = sparse_attention_graph(4096, 384, 4)
    t = time.time()
    print(t-s)
    return g

@tf.function
def dot_circle_product(A, B, G):
    '''
    Returns AB o G, where o is the hadamard product
    G is an adjacency list instead of a matrix, for efficiency
    '''
    n = A.shape[-2]
    m = A.shape[-1]
    p = B.shape[-1]
    print("A:", A.shape)
    print("B:", B.shape)
    print("G:", G.shape)
    
    assert m == B.shape[-2]
    assert n == G.shape[-2]

    result = tf.zeros((n,p))
    for i in tf.range(n):
        for j in G[i]:
            dot_product = tf.reshape(tf.tensordot(A[i,:], B[:,j], axes=1), (1,1))
            paddings = [[i,n-i-1],[j, p-j-1]]
            padded_dot = tf.pad(dot_product, paddings, mode='CONSTANT')
            result += padded_dot
    return result







class MultiHeadSparseAttention(tf.keras.layers.Layer):

    def __init__(self, n_heads, d_model, graph_connections, 
                global_tokens, layer_id, sequence_length):
        super(MultiHeadSparseAttention, self).__init__()
        self.layer_id = layer_id
        self.n_heads = n_heads
        self.d_model = d_model
        self.g_connections = graph_connections
        self.sequence_length = sequence_length
        self.global_tokens = global_tokens
        self.graph = sparse_attention_graph(self.sequence_length, self.g_connections, self.global_tokens)

        assert d_model % n_heads == 0, "Error: n_heads must divide d_model"

        self.depth = d_model // n_heads

        self.wq = tf.keras.layers.Dense(d_model, name='query_weights')
        self.wk = tf.keras.layers.Dense(d_model, name='key_weights')
        self.wv = tf.keras.layers.Dense(d_model, name='value_weights')
        self.dense = tf.keras.layers.Dense(d_model, name='output_weights')
        
    def __str__(self):
        return "Normal Sparse Attention Block " + str(self.layer_id)        

    def split_heads(self, x, batch_size):
        '''
        Split input vectors so so we attend on different parts of them in parallel. eg
        123456789 -> 123 456 789
        '''
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0,2,1,3])

    @tf.function
    def dot_circle_product(self, A, B):
        '''
        Returns AB o self.graph, where o is the hadamard product
        self.graph is an adjacency list instead of a matrix, for efficiency
        '''
        n=A.shape[2]
        G = self.graph
        #print("A:", A.shape)
        #print("B:", B.shape)
        #print("G:", G.shape)
        
        assert A.shape == B.shape
        assert n == G.shape[0]

        result = tf.zeros((A.shape[0], A.shape[1], A.shape[2], A.shape[2]))
        for i in tf.range(n):
            for j in G[i]:
                dot_product = tf.reshape(tf.einsum("abi,abi->ab", A[:,:,i,:], B[:,:,j,:]), (A.shape[0], A.shape[1],1,1))
                paddings = [[0,0],[0,0],[i,n-i-1],[j, n-j-1]]
                padded_dot = tf.pad(dot_product, paddings, mode='CONSTANT')
                result += padded_dot
        return result

    def sparse_attention(self, Q, K, V, mask=None):
        '''
        Computes attention only on the nodes connected by self.graph
        '''
        perm = np.arange(tf.rank(Q))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        qkg = self.dot_circle_product(Q, K)
        dk = tf.cast(K.shape[-1], tf.float32)
        s = qkg / tf.math.sqrt(dk)
        if mask is not None:
            s -= 100000 * mask
        attn_weights = tf.nn.softmax(s, axis=-1)
        output = tf.matmul(attn_weights, V)
        return output, attn_weights

    def call(self, 
            hidden_states, 
            attention_mask=None, 
            layer_head_mask=None, 
            training=False):
        query = hidden_states
        batch_size = query.shape[0]
        seqlen = query.shape[1]
        if seqlen < self.sequence_length:
            paddings = [[0,0],[0,self.sequence_length-seqlen],[0,0]]
            query = tf.pad(query, paddings, "CONSTANT")
        #print(query.shape)

        q = self.wq(query)
        k = self.wk(query)
        v = self.wv(query)

        q = self.split_heads(q, batch_size)
        v = self.split_heads(v, batch_size)
        k = self.split_heads(k, batch_size)

        attention, weights = self.sparse_attention(q, k, v, attention_mask)

        attention = tf.transpose(attention, perm=[0,2,1,3])
        attention = tf.reshape(attention, (batch_size, -1, self.d_model))

        output = self.dense(attention)
        return (output, weights)