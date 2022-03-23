import models.Model as mdl
from models.basic import num_words

from transformers import LEDTokenizer, TFLEDForConditionalGeneration, BertTokenizer
import tensorflow as tf
import numpy as np

LED_MAX_LENGTH = 16384
SAVE_DIR = "models/ckpt/LED"
MNAME = 'allenai/led-base-16384'

class LED(mdl.Model):

    def __init__(self, *args):
        super().__init__()
        
        self.tokenizer = LEDTokenizer.from_pretrained(MNAME)

        
        self.model = TFLEDForConditionalGeneration.from_pretrained(MNAME)

        #self.model = TFLEDForConditionalGeneration()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        self.checkpoint = tf.train.Checkpoint(self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, directory=SAVE_DIR, max_to_keep=3)
        self.ckpt_manager.restore_or_initialize()
        
    def train(self, train, validate, stringify) -> None:
        self.model.compile(
            optimizer=self.optimizer
        )
        self.model.fit(
            x=train,
            batch_size=5,
            epochs=1,
            verbose=2,
            validation_data=validate,
            steps_per_epoch=1,
            validation_steps=1
        )

    def generate(self, data: mdl.DataPair, stringify) -> str:
        input_dict = self.tokenizer(data['article'].numpy().decode('UTF-8'), return_tensors='tf', truncation=True, max_length = LED_MAX_LENGTH, padding='max_length')
        inputs = input_dict['input_ids']
        attn_mask = input_dict['attention_mask']
        global_attention = np.zeros_like(inputs)
        global_attention[:,0] = 1
        global_attention = tf.constant(global_attention)
        prediction = self.model.generate(inputs, global_attention=global_attention, attention_mask=attn_mask, early_stopping=True, max_length=512, num_beams=4)[0]
        #prediction = self.model.generate(inputs, max_length=300, early_stopping=True)[0]
        return self.tokenizer.decode(prediction, skip_special_tokens=True, clean_up_tokenization_spaces=True)


