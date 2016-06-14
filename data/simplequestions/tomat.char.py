import pickle

import numpy as np

from teafacto.util import argprun, tokenize


def run(trainp="fb_train.tsv", testp="fb_test.tsv", validp="fb_valid.tsv", outp="datamat.char.pkl"):
    entdic = {}
    reldic = {}
    chardic = {}
    acc = {}
    acc["train"] = getdata(trainp, chardic, entdic, reldic)
    acc["valid"] = getdata(validp, chardic, entdic, reldic)
    acc["test"] = getdata(testp, chardic, entdic, reldic)
    acc["chardic"] = chardic
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


def getdata(p, chardic, entdic, reldic, maxc=np.infty):
    data = []
    gold = []
    maxlen = 0
    c = 0
    for line in open(p):
        q, a = (line[:-1] if line[-1] == "\n" else line).split("\t")
        s, p = a.split()
        words = tokenize(q)
        q = " ".join(words)
        maxlen = max(maxlen, len(q))
        chars = map(ord, q)
        if len(set(chars).intersection({123})) > 0:
            pass #print line, q
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
    # making chardic and transforming through chardic
    thischardic = dict(map(lambda (x,y): (ord(x), y), chardic.items()))
    nextid = 0
    while nextid in thischardic.values():
        nextid += 1
    uniquechars = np.unique(datamat)
    for uniquechar in list(uniquechars):
        if not uniquechar in thischardic and uniquechar >= 0:
            thischardic[uniquechar] = nextid
            while nextid in thischardic.values():
                nextid += 1
    chardic.update(dict(map(lambda (x, y): (chr(x), y), thischardic.items())))
    print len(chardic), chardic
    datamat = np.vectorize(lambda x: thischardic[x] if x >= 0 else x)(datamat)
    return datamat, goldmat


if __name__ == "__main__":
    argprun(run)