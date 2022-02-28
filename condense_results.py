
import glob
import argparse
import json




OUTPUT_NAME = 'all.json'

def condense(path):
    results = []
    for filename in glob.glob(path+'/*'):
        with open(filename, 'r') as f:
            results += json.load(f)
    for i,x in enumerate(results):
        if x in results[:i]:
            results[i] = 0
    results = [x for x in results if x != 0]
    print(len(results))
    with open(path+OUTPUT_NAME, "w") as f:
        json.dump(results, f)



def parse_me():
    parser = argparse.ArgumentParser(description="Condenser for results from a directory")
    parser.add_argument("path", type=str,
            help="path to the results to condense")
    return parser

if __name__ == '__main__':
    path = parse_me().parse_args().path
    condense(path)


