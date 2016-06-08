import pickle

import numpy as np

from teafacto.util import argprun


def run(trainp="fb_train.tsv", testp="fb_test.tsv", validp="fb_valid.tsv", outp="datamat.char.pkl"):
    entdic = {}
    reldic = {}
    acc = {}
    acc["train"] = getdata(trainp, entdic, reldic)
    acc["valid"] = getdata(validp, entdic, reldic)
    acc["test"] = getdata(testp, entdic, reldic)
    numents = len(entdic)
    reldic = {k: v+numents for k, v in reldic.items()}
    entdic.update(reldic)
    print len(entdic)
    acc["entdic"] = entdic
    acc["numents"] = numents
    pickle.dump(acc, open(outp, "w"))


def getdata(p, entdic, reldic, maxc=np.infty):
    data = []
    gold = []
    maxlen = 0
    c = 0
    for line in open(p):
        q, a = (line[:-1] if line[-1] == "\n" else line).split("\t")
        s, p = a.split()
        maxlen = max(maxlen, len(q))
        chars = map(ord, q)
        if s not in entdic:
            entdic[s] = len(entdic)
        if p not in reldic:
            reldic[p] = len(reldic)
        data.append(chars)
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