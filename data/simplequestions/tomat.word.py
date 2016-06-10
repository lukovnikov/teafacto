import pickle, re

import numpy as np

from teafacto.util import argprun, tokenize


def run(trainp="fb_train.tsv", testp="fb_test.tsv", validp="fb_valid.tsv", outp="datamat.word.pkl"):
    worddic = {"<RARE>": 0}
    entdic = {}
    reldic = {}
    acc = {}
    acc["train"] = getdata(trainp, worddic, entdic, reldic)
    acc["valid"] = getdata(validp, worddic, entdic, reldic)
    acc["test"] = getdata(testp, worddic, entdic, reldic)
    acc["worddic"] = worddic
    numents = len(entdic)
    acc["train"][1][:, 1] += numents
    acc["valid"][1][:, 1] += numents
    acc["test"][1][:, 1] += numents
    reldic = {k: v+numents for k, v in reldic.items()}
    entdic.update(reldic)
    print len(entdic)
    acc["entdic"] = entdic
    acc["numents"] = numents
    pickle.dump(acc, open(outp, "w"))


def getwords(s):
    return tokenize(s)


def getdata(p, worddic, entdic, reldic, maxc=np.infty):
    data = []
    gold = []
    maxlen = 0
    c = 0
    for line in open(p):
        q, a = (line[:-1] if line[-1] == "\n" else line).split("\t")
        s, p = a.split()
        words = getwords(q)
        maxlen = max(maxlen, len(words))
        for word in words:
            if word not in worddic:
                worddic[word] = len(worddic)
        if s not in entdic:
            entdic[s] = len(entdic)
        if p not in reldic:
            reldic[p] = len(reldic)
        wordidx = map(lambda x: worddic[x], words)
        data.append(wordidx)
        gold.append([entdic[s], reldic[p]])
        c += 1
        if c > maxc:
            break
    datamat = np.zeros((c, maxlen)).astype("int32") - 1
    goldmat = np.zeros((c, 2)).astype("int32")
    i = 0
    for x in data:
        datamat[i, :len(x)] = x
        i += 1
    i = 0
    for x in gold:
        goldmat[i, :] = x
        i += 1
    return datamat, goldmat


if __name__ == "__main__":
    argprun(run)
    #getwords("What's is plaza-midwood (wood) a type of?")