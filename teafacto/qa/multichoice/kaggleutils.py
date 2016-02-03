import os, pandas as pd
from nltk.tokenize import RegexpTokenizer

def read(path="../../../data/kaggleai/training_set.tsv"):
    path = os.path.join(os.path.dirname(__file__), path)
    file = open(path)
    todf = []
    first = True
    cols = None
    for line in file:
        ls = line[:-1].split("\t")
        if first:
            cols = ls
            first = False
            continue
        ls = [int(ls[0])] + [tokenize(x) for x in ls[1:]]
        todf.append(ls)
    df = pd.DataFrame(data=todf, columns=cols)
    df.index = df["id"]
    del df["id"]
    return df

def tokenize(ll):
    tokens = RegexpTokenizer(r'\w+').tokenize(ll)
    if len(tokens) == 1 and len(tokens[0]) == 1: # one letter:
        return tokens[0]
    else:
        return map(lambda x: x.lower(), tokens)