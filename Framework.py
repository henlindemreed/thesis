'''
Henry Lindeman

The basic functions that will let me actually test stuff
'''
import models.Model as Mdl
from rouge_metric import PyRouge
import glob
import random
import json
from typing import Tuple

import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np 
import pandas as pd 

# Load a dataset in the path or identifier given
# * dataset_name: the path to or identifier of the dataset
# * returns: A list of all the data in the dataset
def load_dataset(dataset_name: str) -> Mdl.Paperset:
    if dataset_name in ('../textinfo', 'sysconf'):
        filenames = glob.glob('../../textinfo/jsons/*.json')
        random.shuffle(filenames)
        data = [None] * len(filenames)
        for i in range(len(filenames)):
            filename = filenames[i][18:-5]
            with open(filenames[i], "r") as f:
                x = json.load(f)
                data[i] = Mdl.DataPair(id=filename, abstr=x["abstract"], paper=x["paper"])
        return data

    raise NotImplementedError("I don't know about dataset " + dataset_name)


# Split a dataset into two parts: a training and a testing set
# * D: the dataset to split
# * eval_pct: the percentage of datapoints to test rather than train on
# * returns: <Training data, Testing data>
def partition_dataset(D: Mdl.Paperset, eval_pct: float) -> Tuple[Mdl.Paperset, Mdl.Paperset]:
    num_to_test = int(eval_pct * len(D))
    testing = random.sample(D, num_to_test)
    training = [d for d in D if d not in testing]
    return training, testing


# Train a model on training data
# * M: the model to be trained
# * D: the training data
def Train(M: Mdl.Model, D: Mdl.Paperset) -> None:
    M.train(D)


# Evaluate a trained model on testing data, with ROUGE scores
# * M: the model to be evaluated
# * D: the testing data
# * returns: A list matching paper titles to how well the model scored on their abstracts
def Evaluate(M: Mdl.Model, D: Mdl.Paperset):
    rouge = PyRouge()
    results = []
    for d in D:
        results.append((d.id, rouge.evaluate([M.generate(d)], [[d.abstr]]) ))
    return results


# Builds the model requested by importing and constructing
# * model_name: name of a model
# * n         : a parameter n used by some models
# * returns: the model
def init_model(model_name: str, n: int = 2) -> Mdl.Model:
    if model_name == "nonsense":
        from models.nonsense import Nonsense
        return Nonsense()
    if model_name == "cheater":
        from models.cheater import Cheater
        return Cheater()
    if model_name == "basic":
        from models.basic import Basic
        return Basic()
    if model_name == "stochastic_ngram":
        from models.stochasticngram import StochasticNGram
        return StochasticNGram(n)
    if model_name in ("LED", "Longformer"):
        from models.LED import LED
        return LED()
    raise NotImplementedError("I don't know about model " + model_name)


# Performs an entire test on a model
# * model_name: the model to test
# * dataset_name: the dataset to test on
# * eval_pct: the percent of data to use for evaluation
# * returns: A list matching paper titles to how well the model scored on their abstracts
def test_model(model_name: str, dataset_name: str, eval_pct: float):
    training, testing = partition_dataset(load_dataset(dataset_name), eval_pct)
    print("----------- LOADED DATA ------------")
    M = init_model(model_name)
    print("----------- LOADED MODEL -----------")
    Train(M, training)
    print("---------- TRAINED MODEL -----------")
    return Evaluate(M, testing)

def load_arxiv() -> tuple:
    db = tfds.load('scientific_papers')
    db.download_and_prepare()
    ds = db.as_dataset()
    return ds['train'], ds['test'], ds['validation']


if __name__ == "__main__":
    result = test_model("Longformer", "sysconf", 0.10)
    print("--------- EVALUATED MODEL ----------")
    with open("LED_result.json", "w") as f:
        json.dump(result, f)