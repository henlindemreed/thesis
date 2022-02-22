import models.Model as mdl

import tensorflow as tf
import tensorflow_text as tft

DATASET = 'arxiv'
PEGASUS = True

tf.compat.v1.enable_resource_variables()

def compute_path(dataset, pegasus):
    if pegasus:
        if dataset=='arxiv':
            return "bigbird/ckpt/summarization/arxiv/pegasus/saved_model"
        if dataset=='pubmed':
            return "bigbird/ckpt/summarization/pubmed/pegasus/saved_model"
    else:
        if dataset=='arxiv':
            return "bigbird/ckpt/summarization/arxiv/roberta/saved_model"
        if dataset=='pubmed':
            return "bigbird/ckpt/summarization/pubmed/roberta/saved_model"
    raise ValueError("dataset " + str(dataset) + " is unrecognized")


class BigBird(mdl.Model):

    def __init__(self, *args):
        super().__init__()
        path = compute_path(DATASET, PEGASUS)
        self.model = tf.saved_model.load(path, tags='serve')

    def train(self, data: mdl.Paperset) -> None:
        pass

    def generate(self, data: mdl.DataPair, stringify) -> str:
        summarize = self.model.signatures['serving_default']
        candidate = summarize(data['article'])
        return candidate['pred_sent'][0].numpy().decode('UTF-8')
