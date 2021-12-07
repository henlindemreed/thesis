import models.Model as mdl

from transformers import PegasusTokenizer, BigBirdPegasusForConditionalGeneration, BigBirdPegasusConfig
import torch

class BigBirdPegasus(mdl.Model):

    def __init__(self, *args):
        super().__init__()
        self.tokenizer = PegasusTokenizer.from_pretrained('bigbird-pegasus-large-arxiv')
        self.model = BigBirdPegasusForConditionalGeneration.from_pretrained('bigbird-pegasus-large-arxiv')

    def train(self, data: mdl.Paperset) -> None:
        pass

    def generate(self, data: mdl.DataPair) -> str:
        inputs = self.tokenizer([data.paper], max_length=4096, return_tensors='pt', truncation=True)['input_ids']
        prediction = self.model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)[0]
        return ' '.join( [ self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in prediction] )
