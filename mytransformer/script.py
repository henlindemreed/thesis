from re import M
from Mytransformer import MyTransformer
from attention import *

'''
A = tf.cast(tf.random.uniform((8,100), minval=0, maxval=400, dtype=tf.int32), dtype=tf.float32)
G = sparse_attention_graph(8,4)
print(tf.matmul(A,tf.transpose(A, perm=[1,0])))
print(G)
print(sparse_attention(A,A,A,G))
'''

m = MyTransformer.from_pretrained('facebook/bart-large')