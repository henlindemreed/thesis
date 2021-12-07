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


Paperset = List[DataPair]

class Model(ABC):

    def __init__(self, *args):
        pass

    @abstractmethod
    def train(self, data: Paperset) -> None:
        pass

    @abstractmethod
    def generate(self, data: DataPair) -> str:
        pass