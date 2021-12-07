'''
A model that cheats by looking at the abstract before outputting a guess at the abstract
'''
import models.Model as Mdl
from nltk.tokenize import word_tokenize
import random

START = "<#START#>"
STOP = "<#STOP#>"

# Here's a dictionary that an count easy and combine with other counters
class counter(dict):

    def count(self, key):
        if key in self:
            self[key] += 1
        else:
            self[key] = 1

    def __add__(self, other):
        ans = counter()
        for k in self:
            ans[k] = self[k]
        for k in other:
            if k in ans:
                ans[k] += other[k]
            else:
                ans[k] = other[k]
        return ans

# Return the count for each n-gram in a counter obj
def parse_text(text, n):
    ans = counter()
    words = [START] * (n-1) + word_tokenize(text) + [STOP] * (n-1)
    for i in range(len(words)-n+1):
        ans.count(tuple(words[i:i+n]))
    return ans

# normalize each list of counts to probabilities
def normalize(stats):
    ans = {}
    for g in stats:
        l = stats[g][1].copy()
        s = sum(l)
        for i in range(len(l)):
            l[i] = float(l[i]) / s
        ans[g] = (stats[g][0], l)
    return ans

def merge(stats1, stats2, binop=lambda x, y: x+y):
    ans = {}
    for gram in set(stats1.keys()) | set(stats2.keys()):
        if gram in stats1 and gram in stats2:
            s1_words = stats1[gram][0]
            s1_cts = stats1[gram][1]
            s2_words = stats2[gram][0]
            s2_cts = stats2[gram][1]
            words = list(set(s1_words) | set(s2_words))
            cts = []
            for w in words:
                s1_num, s2_num = 0,0
                try:
                    s1_num = s1_cts[s1_words.index(w)]
                except ValueError:
                    cts.append(s2_cts[s2_words.index(w)])
                try:
                    s2_num = s2_cts[s2_words.index(w)]
                except ValueError:
                    cts.append(s1_cts[s1_words.index(w)])
                if s1_num * s2_num != 0:
                    cts.append(binop(s1_num, s2_num))
            ans[gram] = (words, cts)
                
        elif gram in stats1:
            ans[gram] = stats1[gram]
        else:
            ans[gram] = stats2[gram]
    return ans

class StochasticNGram(Mdl.Model):

    def __init__(self, n=2, *args):
        self.n = n
        self.trained = False

    def train(self, data):
        # count the n-grams in training texts
        universal_paper_grams = counter()
        universal_abstr_grams = counter()
        for d in data:
            universal_abstr_grams = universal_abstr_grams + parse_text(d.abstr, self.n)
            universal_paper_grams = universal_paper_grams + parse_text(d.paper, self.n)

        # collect into stochastic next-token tables
        self.next_paper_grams = {}
        for gram in universal_paper_grams:
            if gram[:-1] in self.next_paper_grams:
                self.next_paper_grams[gram[:-1]][0].append(gram[-1])
                self.next_paper_grams[gram[:-1]][1].append(universal_paper_grams[gram])
            else:
                self.next_paper_grams[gram[:-1]] = ([gram[-1]], [universal_paper_grams[gram]])

        self.next_abstr_grams = {}
        for gram in universal_abstr_grams:
            if gram[:-1] in self.next_abstr_grams:
                self.next_abstr_grams[gram[:-1]][0].append(gram[-1])
                self.next_abstr_grams[gram[:-1]][1].append(universal_abstr_grams[gram])
            else:
                self.next_abstr_grams[gram[:-1]] = ([gram[-1]], [universal_abstr_grams[gram]])
        
        # compute average abstract length
        self.abstrlen = sum([ sum(v[1]) for v in self.next_abstr_grams.values()]) / len(data)
        self.trained = True


    def generate(self, data):
        # first, make sure that the model has been trained
        if not self.trained:
            raise ValueError("Generation attempted on an empty model. Please give some data to train on")

        # compute this paper's stats like in training
        specific_paper_grams = parse_text(data.paper, self.n)
        next_specific_grams = {}
        for gram in specific_paper_grams:
            if gram[:-1] in next_specific_grams:
                next_specific_grams[gram[:-1]][0].append(gram[-1])
                next_specific_grams[gram[:-1]][1].append(specific_paper_grams[gram])
            else:
                next_specific_grams[gram[:-1]] = ([gram[-1]], [specific_paper_grams[gram]])

        # merge this paper's stats into all paper's stats and normalize
        all_next_grams = normalize(merge(merge(next_specific_grams, self.next_abstr_grams),
                                        self.next_paper_grams))
        pre_next_grams = normalize(merge(normalize(next_specific_grams), normalize(self.next_abstr_grams), max))

        candidate = [START] * (self.n-1)
        while len(candidate) < self.abstrlen + 2 and candidate[-1] != STOP:
            gram = tuple(candidate[1-self.n:])
            words = pre_next_grams[gram][0]
            all_words = all_next_grams[gram][0]
            weights = [pre_prob / all_next_grams[gram][1][all_words.index(w)]
                        for w, pre_prob in zip(words, pre_next_grams[gram][1])]
            next_word = random.choices(words, weights=weights, k=1)[0]
            candidate.append(next_word)
        return " ".join(candidate[self.n:-1])