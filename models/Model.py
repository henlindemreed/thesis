'''
Henry Lindeman

Abstract python class for an abstractive text-summarization model
'''

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

@dataclass
class DataPair:
    id: str
    abstr: str
    paper: str

    def __eq__(self, other):
        return self.id == other.id

    def __getitem__(self, key):
        if key == 'abstract':
            return self.abstr
        if key in ('paper', 'article'):
            return self.paper
        if key == 'id':
            return self.id
        raise KeyError("The key \"" + str(key) + "\" is an invalid key for this object. Available are: \"abstract\", \"paper\", and \"id\"")

Paperset = List[DataPair]

class Model(ABC):

    def __init__(self, *args):
        pass

    @abstractmethod
    def train(self, data: Paperset, stringify) -> None:
        pass

    @abstractmethod
    def generate(self, data: DataPair, stringify) -> str:
        pass
