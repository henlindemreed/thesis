'''
A model that picks the first n words, where n is the average number of words in abstracts
'''

import models.Model as Mdl
import tensorflow as tf

def num_words(x: str) -> int:
    return len(tf.strings.split(x))

class Basic(Mdl.Model):
    def __init__(self, *args):
        self.avg_len = 5

    def train(self, data, stringify):
        total_words = sum([len(stringify(d['abstract']).split()) for d in data])
        self.avg_len = total_words // len(data)

    def generate(self, data, stringify):
        return tf.strings.join(stringify(data['article']).split()[:self.avg_len], separator=' ')
