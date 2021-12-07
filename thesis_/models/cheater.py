'''
A model that cheats by looking at the abstract before outputting a guess at the abstract
'''
import models.Model as Mdl


class Cheater(Mdl.Model):

    def __init__(self, *args):
        pass

    def train(self, data):
        pass

    def generate(self, data):
        return data.abstr