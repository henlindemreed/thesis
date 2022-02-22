'''
Henry Lindeman

The basic functions that will let me actually test stuff
'''
import models.Model as Mdl
from utils import *

from rouge_metric import PyRouge
import glob
import random
import datetime
import json
from typing import Tuple

import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np 
import pandas as pd 

# Load a dataset in the path or identifier given
# * dataset_name: the path to or identifier of the dataset
# * returns: (Training, Testing, Validation) sets
def load_dataset(dataset_name: str, eval_pct: float):
    
    if dataset_name in ('../textinfo', 'sysconf'):
        filenames = glob.glob('../textinfo/jsons/*.json')
        random.seed(1)
        random.shuffle(filenames)
        data = [None] * len(filenames)
        for i in range(len(filenames)):
            filename = filenames[i][18:-5]
            with open(filenames[i], "r") as f:
                x = json.load(f)
                data[i] = Mdl.DataPair(id=filename, abstr=x["abstract"], paper=x["paper"])
        return partition_dataset(data, eval_pct)
    
    if dataset_name in ('arxiv', 'scientific_papers'):
        return load_arxiv()

    raise NotImplementedError("I don't know about dataset " + dataset_name)


# Split a dataset into two parts: a training and a testing set
# * D: the dataset to split
# * eval_pct: the percentage of datapoints to test rather than train on
# * returns: <Training data, Testing data>
def partition_dataset(D: Mdl.Paperset, eval_pct: float) -> Tuple[Mdl.Paperset, Mdl.Paperset, Mdl.Paperset]:
    num_to_test = int(eval_pct * len(D))
    random.seed(0)
    testing = random.sample(D, num_to_test)
    training_plus_validation = [d for d in D if d not in testing]
    validation = random.sample(training_plus_validation, num_to_test)
    training = [d for d in training_plus_validation if d not in validation]
    return training, testing, validation, lambda x:x


# Train a model on training data
# * M: the model to be trained
# * D: the training data
def Train(M: Mdl.Model, D: Mdl.Paperset, stringify, drop_me) -> None:
    M.train(D, stringify)


# Evaluate a trained model on testing data, with ROUGE scores
# * M: the model to be evaluated
# * D: the testing data
# * returns: A list matching paper titles to how well the model scored on their abstracts
def Evaluate(M: Mdl.Model, D: Mdl.Paperset, stringify, drop_me):
    rouge = PyRouge()
    length = len(D)
    i = 0
    markers = [int(x * length / 100) for x in range(100)]
    results = []
    for d in D:
        i += 1
        candidate = M.generate(d, stringify)
        results.append((stringify(d['id']), rouge.evaluate([candidate], [[stringify(d['abstract'])]]) ))
        if i in markers:
            print(str(datetime.datetime.now().time()) + ': -------------- ' + str(100*i / length) + '% ---------------')
    return results, 1.0


# Builds the model requested by importing and constructing
# * model_name: name of a model
# * n         : a parameter n used by some models
# * returns: the model
def init_model(model_name: str, stringify, n: int = 2) -> Mdl.Model:
    if model_name == "nonsense":
        from models.nonsense import Nonsense
        return Nonsense(), lambda x: False
    if model_name == "cheater":
        from models.cheater import Cheater
        return Cheater(), lambda x: False
    if model_name == "basic":
        from models.basic import Basic
        return Basic(), lambda x: False
    if model_name == "stochastic_ngram":
        from models.stochasticngram import StochasticNGram
        return StochasticNGram(n), lambda x: False
    if model_name in ("LED", "Longformer"):
        from models.LED import LED, LED_MAX_LENGTH
        return LED(), lambda x: (len(stringify(x).split()) > LED_MAX_LENGTH)
    if model_name == "BigBird":
        from models.BigBird import BigBird
        return BigBird(), lambda x: False
    raise NotImplementedError("I don't know about model " + model_name)


# Performs an entire test on a model
# * model_name: the model to test
# * dataset_name: the dataset to test on
# * eval_pct: the percent of data to use for evaluation
# * returns: A list matching paper titles to how well the model scored on their abstracts
def test_model(model_name: str, dataset_name: str, eval_pct: float):
    training, testing, validation, stringify = load_dataset(dataset_name, eval_pct)
    print("----------- LOADED DATA ------------")
    M, drop_me = init_model(model_name, stringify)
    print("----------- LOADED MODEL -----------")
    Train(M, training, stringify, drop_me)
    print("---------- TRAINED MODEL -----------")
    result, yeild = Evaluate(M, testing, stringify, drop_me)
    print("-------- EVALUATED MODEL -----------")
    print(str(yeild*100) + "% of papers were short enough to generate a candidate summary")
    return result

def load_arxiv() -> tuple:
    ds = tfds.load('scientific_papers')
    return ds['train'], ds['test'], ds['validation'], lambda x: x.numpy().decode('UTF-8')


if __name__ == "__main__":
    result = test_model("LED", "arxiv", 0.10)
    filename = "longformer_arxiv_result.json"
    print("results will be found in " + filename)
    with open(filename, "w") as f:
        json.dump(result, f)



