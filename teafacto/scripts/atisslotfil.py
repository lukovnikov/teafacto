import pickle
from teafacto.util import argprun

def run(p="../../data/atis/atis.pkl"):
    train, test, dics = pickle.load(open(p))
    word2idx = dics["words2idx"]
    table2idx = dics["tables2idx"]
    label2idx = dics["labels2idx"]
    train = zip(*train)
    test = zip(*test)
    print len(train)
    print len(test)
    tup2text(train[0], word2idx, table2idx, label2idx)
    maxlen = 0
    for tup in train + test:
        maxlen = max(len(tup[0]), maxlen)
    print maxlen

def tup2text(tup, word2idx, table2idx, label2idx):
    word2idxrev = {v: k for k, v in word2idx.items()}
    table2idxrev = {v: k for k, v in table2idx.items()}
    label2idxrev = {v: k for k, v in label2idx.items()}
    i = 0
    words = " ".join(map(lambda x: word2idxrev[tup[0][x]], range(len(tup[0]))))
    labels = " ".join(map(lambda x: label2idxrev[tup[2][x]], range(len(tup[0]))))
    print words
    print labels


if __name__ == "__main__":
    argprun(run)