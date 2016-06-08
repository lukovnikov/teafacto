

import pickle

import numpy as np

from teafacto.util import argprun


def run(trainp="fb_train.tsv", testp="fb_test.tsv", validp="fb_valid.tsv", outp="datamat.word.pkl", maxchar=70):
    worddic = {"<RARE>": 0}
    entdic = {}
    reldic = {}
    acc = {}
    acc["train"] = getdata(trainp, worddic, entdic, reldic, maxchar=maxchar)
    acc["valid"] = getdata(validp, worddic, entdic, reldic, maxchar=maxchar)
    acc["test"] = getdata(testp, worddic, entdic, reldic, maxchar=maxchar)
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


def getdata(p, worddic, entdic, reldic, maxc=np.infty, maxchar=50):
    data = []
    gold = []
    maxlen = 0
    c = 0
    for line in open(p):
        q, a = (line[:-1] if line[-1] == "\n" else line).split("\t")
        s, p = a.split()
        words = q.split()
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
    datamat = np.zeros((c, maxlen, maxchar+1)).astype("int32") - 1
    goldmat = np.zeros((c, 2)).astype("int32")
    revworddic = {v: k for k, v in worddic.items()}
    i = 0
    for x in data:
        j = 0
        for xe in x:
            datamat[i, j, 0] = xe
            chars = map(ord, revworddic[xe])
            datamat[i, j, 1:len(chars)+1] = chars
            j += 1
        i += 1
    i = 0
    for x in gold:
        goldmat[i, :] = x
        i += 1
    return datamat, goldmat


if __name__ == "__main__":
    argprun(run)