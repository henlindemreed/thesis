'''
A model that picks the first n words, where n is the average number of words in abstracts
'''

import models.Model as Mdl

def num_words(x: str) -> int:
    return len(x.split())

class Basic(Mdl.Model):
    def __init__(self, *args):
        self.avg_len = 5

    def train(self, data):
        total_words = sum([num_words(d.abstr) for d in data])
        self.avg_len = total_words // len(data)

    def generate(self, data):
        return ' '.join(data.paper.split()[:self.avg_len])