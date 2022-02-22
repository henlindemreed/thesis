'''
A generation model that produces a constant nonsense-string
'''
import models.Model as Mdl

NONSENSE = "kjdshf kjsahf d oeiuhdkjc nqwe hxnjklepoh doweu f fliew hfqcksab; od  w fasdjaw kjehbfqe jfnwkej ewf wef ew fj wdcaisjfc krnc cnwei qhdwoqlsnzncue\
lk lij rfqw, dm slk w3 ids slc90 ksndacs; s dsdl jf cscd, zef lk. lkad fsdmsae jdksawoi  wdzi sknfvbr r riuac sskfhskdj   dfjdh  f sks kd  oiie jklsakfeuhq q."

class Nonsense(Mdl.Model):

    def __init__(self, *args):
        pass

    def train(self, data, stringify):
        pass

    def generate(self, data, stringify):
        return NONSENSE
