import pickle

import numpy as np

from teafacto.util import argprun, tokenize


def run(trainp="fb_train.tsv", testp="fb_test.tsv", validp="fb_valid.tsv", outp="datamat.wordchar.pkl", maxchar=30):
    worddic = {"<RARE>": 0}
    chardic = {}
    entdic = {}
    reldic = {}
    acc = {}
    acc["train"] = getdata(trainp, worddic, chardic, entdic, reldic, maxchar=maxchar)
    acc["valid"] = getdata(validp, worddic, chardic, entdic, reldic, maxchar=maxchar)
    acc["test"] = getdata(testp, worddic, chardic, entdic, reldic, maxchar=maxchar)
    acc["worddic"] = worddic
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


def getdata(p, worddic, chardic, entdic, reldic, maxc=np.infty, maxchar=30):
    data = []
    gold = []
    maxlen = 0
    maxwordlen = 0
    c = 0
    for line in open(p):
        q, a = (line[:-1] if line[-1] == "\n" else line).split("\t")
        s, p = a.split()
        words = tokenize(q)
        maxlen = max(maxlen, len(words))
        for word in words:
            maxwordlen = max(maxwordlen, len(word))
            if word not in worddic:
                worddic[word] = len(worddic)
        if s not in entdic:
            entdic[s] = len(entdic)
        if p not in reldic:
            reldic[p] = len(reldic)
        data.append(words)
        gold.append([entdic[s], reldic[p]])
        c += 1
        if c > maxc:
            break
    print maxwordlen
    maxchar = min(maxchar, maxwordlen)
    wordmat = np.zeros((c, maxlen)).astype("int32") - 1
    charten = np.zeros((c, maxlen, maxchar)).astype("int32") - 1
    goldmat = np.zeros((c, 2)).astype("int32")
    i = 0
    for sent in data:
        j = 0
        for word in sent:
            if len(word) > maxchar:
                print word
            wordmat[i, j] = worddic[word]
            chars = map(ord, word)
            charten[i, j, :min(len(chars), maxchar)] = chars[:min(len(chars), maxchar)]
            j += 1
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
    uniquechars = np.unique(charten)
    for uniquechar in list(uniquechars):
        if not uniquechar in thischardic and uniquechar >= 0:
            thischardic[uniquechar] = nextid
            while nextid in thischardic.values():
                nextid += 1
    chardic.update(dict(map(lambda (x, y): (chr(x), y), thischardic.items())))
    print len(chardic), chardic
    charten = np.vectorize(lambda x: thischardic[x] if x >= 0 else x)(charten)
    datamat = np.concatenate([wordmat.reshape(wordmat.shape + (1,)), charten], axis=2)
    return datamat, goldmat


if __name__ == "__main__":
    argprun(run)