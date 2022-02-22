import models.Model as mdl
from models.basic import num_words

from transformers import LEDTokenizer, TFLEDForConditionalGeneration
import tensorflow as tf

LED_MAX_LENGTH = 16000

class LED(mdl.Model):

    def __init__(self, *args):
        super().__init__()
        self.tokenizer = LEDTokenizer.from_pretrained('allenai/led-large-16384-arxiv')
        self.model = TFLEDForConditionalGeneration.from_pretrained('allenai/led-large-16384-arxiv')

    def train(self, data: mdl.Paperset, stringify) -> None:
        pass

    def generate(self, data: mdl.DataPair, stringify) -> str:
        inputs = self.tokenizer.encode(data['article'].numpy().decode('UTF-8'), return_tensors='tf', truncation=True, max_length = 16384)
        prediction = self.model.generate(inputs, num_beams=3, max_length=150, early_stopping=True, experimental_relax_shape=True)[0]
        return self.tokenizer.decode(prediction, skip_special_tokens=True, clean_up_tokenization_spaces=True)


