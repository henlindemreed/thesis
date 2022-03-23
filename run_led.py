# For Arxiv

import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import LEDTokenizer
from models.LED import LED, LED_MAX_LENGTH
import numpy as np

tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)

BATCH_SIZE = 5


def main():
    ds = tfds.load("scientific_papers")
    train, test, validate = ds['train'], ds['test'], ds['validation']

    model = LED()

    #train = train.batch(BATCH_SIZE)

    tokenizer = model.tokenizer
    '''
    arts = train.map(lambda x: x['article'])
    abst = train.map(lambda x: x['abstract'])
    valart = validate.map(lambda x: x['article'])
    valabs = validate.map(lambda x: x['abstract'])

    print(abst)
    '''
    #for e in abst.take(1):
        #print(e)
    print('==============')

    def tf_to_string(tfstring):
        return tf.compat.as_str_any(tfstring)

    def tokenize(example):
        art = tokenizer(
            tf_to_string(example['article']), 
            return_tensors='tf', truncation=True, 
            max_length=LED_MAX_LENGTH, padding='max_length')
        art_in = tf.reshape(art['input_ids'], [LED_MAX_LENGTH])
        att_mask = tf.reshape(art['attention_mask'], [LED_MAX_LENGTH])
        abs = tokenizer(
            tf_to_string(example['abstract']),
            return_tensors='tf', truncation=True, 
            max_length=LED_MAX_LENGTH, padding='max_length')
        abs_in = tf.reshape(abs['input_ids'], [LED_MAX_LENGTH])
        return {'input_ids': art_in, 'attention_mask': att_mask, 'labels': abs_in}
    
    train = train.map(tokenize).shuffle(1000).batch(BATCH_SIZE)
    validate = validate.map(tokenize).shuffle(1000).batch(BATCH_SIZE)  
    
    model.train(
        train=train,
        validate=validate,
        stringify=None
    )

if __name__ == '__main__':
    main()
    