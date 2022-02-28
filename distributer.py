import argparse
import subprocess
import random
from Framework import load_dataset







def parse_me():
    parser = argparse.ArgumentParser(description="Distributer for minion processing")
    parser.add_argument("dataset_name", type=str,
            help="name of the summarization dataset")
    parser.add_argument("model_name", type=str,
            help="name of the summarizer model")
    parser.add_argument("nproc", type=int,
            help="how many minions to spawn")
    parser.add_argument("yeild", type=float,
            help="what proportion of the dataset to process")
    parser.add_argument("outdir", type=str, 
            help="directory containing the output files")
    return parser

if __name__ == '__main__':
    args = parse_me().parse_args()
    _, te, _, _ = load_dataset(args.dataset_name, 0.10)
    to_process = int(len(te) * args.yeild)
    window_size = to_process // args.nproc
    nwindows = len(te) // window_size
    window_choices = random.sample(range(nwindows), args.nproc)
    commands = ['nice', 'python3', 'minion.py', args.model_name, args.dataset_name, 0, str(window_size), ""]
    for w in window_choices:
        outfile = args.outdir+"/"+args.dataset_name+"/"+args.model_name+"/"+str(w)+".json"
        commands[5] = str(w)
        commands[7] = outfile
        print(" ".join(commands))
        task = subprocess.Popen(commands)
        # Comment this out for parallel tasks
        task.communicate()











