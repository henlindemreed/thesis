import json
from math import sqrt



def load(path):
    with open(path, 'r') as f:
        return json.load(f)

def mean_variance(scores):
    r1_tot, r2_tot, rl_tot = 0,0,0
    for s in scores:
        r1_tot += s[1]['rouge-1']['f']
        r2_tot += s[1]['rouge-2']['f']
        rl_tot += s[1]['rouge-l']['f']
    r1_mean = r1_tot / len(scores)
    r2_mean = r2_tot / len(scores)
    rl_mean = rl_tot / len(scores)
    r1_var, r2_var, rl_var = 0,0,0
    for s in scores:
        r1_var += (s[1]['rouge-1']['f'] - r1_mean)**2
        r2_var += (s[1]['rouge-2']['f'] - r2_mean)**2
        rl_var += (s[1]['rouge-l']['f'] - rl_mean)**2
    r1_var /= len(scores)
    r2_var /= len(scores)
    rl_var /= len(scores)
    r1_var = sqrt(r1_var)
    r2_var = sqrt(r2_var)
    rl_var = sqrt(rl_var)
    return {
        'mean': {
            'rouge-1': r1_mean,
            'rouge-2': r2_mean,
            'rouge-l': rl_mean
        }, 'variance': {
            'rouge-1': r1_var,
            'rouge-2': r2_var,
            'rouge-l': rl_var
        }
    }

if __name__ == '__main__':
    path = 'results/arxiv/BigBird/all.json'
    print(path)
    crunched = mean_variance(load(path))
    for k1 in crunched:
        for k2 in crunched[k1]:
            print(k2, k1, str(crunched[k1][k2]))