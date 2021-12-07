import models.Model as mdl

from transformers import LEDTokenizer, LEDForConditionalGeneration
import torch

class LED(mdl.Model):

    def __init__(self, *args):
        super().__init__()
        self.tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
        self.model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')

    def train(self, data: mdl.Paperset) -> None:
        pass

    def generate(self, data: mdl.DataPair) -> str:
        inputs = self.tokenizer([data.paper], return_tensors='pt')['input_ids']
        global_attention_mask = torch.zeros_like(inputs)
        global_attention_mask[:,0] = 1
        prediction = self.model.generate(inputs, global_attention_mask=global_attention_mask,
                            num_beams=3, max_length=150, early_stopping=True)[0]
        return self.tokenizer.decode(prediction, skip_special_tokens=True, clean_up_tokenization_spaces=True)
