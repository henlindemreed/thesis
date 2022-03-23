
from models.Model import DataPair
from Framework import *
import argparse

import json



def parser():
    parser = argparse.ArgumentParser(description="Process a subset of a dataset with some model")
    parser.add_argument("model_name",   type=str,
            help="Name of the model to use for summarization")
    parser.add_argument("dataset_name", type=str,
            help="Name of the summarization dataset")
    parser.add_argument("window_number",   type=int,
            help="Window (of size <NUMBER>) to start at in the dataset")
    parser.add_argument("len_data",  type=int, 
            help="How many data to process")
    parser.add_argument("outfile", type=argparse.FileType('w', encoding='UTF-8'), 
            help="Where to write my results")
    return parser

if __name__ == '__main__':
    args = parser().parse_args()
    tr, te, va, stringify = load_dataset(args.dataset_name, 0.10)
    model, drop_me = init_model(args.model_name, stringify)
    
    '''
    # Use me for sysconf!
    segment_start = args.window_number * args.len_data
    segment = te[segment_start: segment_start + args.len_data]
    result, yeild = Evaluate(model, segment, stringify, drop_me)
    print("Yeild for window {}: {}%".format(args.window_number, yeild * 100))
    json.dump(result, args.outfile)
    
    '''
    # Use me for arxiv!
    window = None
    i = 0
    for w in te.window(args.len_data, shift=args.len_data):
        if i == args.window_number:
            to_process = tf.data.Dataset.zip((w['article'],w['abstract'],w['section_names'])).map(lambda x, y, z: {"article":x, "abstract":y, "id":z})
            
            result, yeild = Evaluate(model, to_process, stringify, drop_me)
            print("Yeild for window {}: {}%".format(args.window_number, yeild * 100))
            json.dump(result, args.outfile)
            break
        i += 1
    

