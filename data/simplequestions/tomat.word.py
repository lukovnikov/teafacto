import pickle

import numpy as np

from teafacto.datahelp.labelsearch import SimpleQuestionsLabelIndex
from teafacto.util import argprun, tokenize


def run(trainp="fb_train.tsv",
        testp="fb_test.tsv",
        validp="fb_valid.tsv",
        outp="datamat.word.dmp.pkl",
        dmp=True):
    worddic = {"<RARE>": 0}
    entdic = {}
    reldic = {}
    acc = {}
    idx = None
    if dmp:
        idx = SimpleQuestionsLabelIndex(host="drogon", index="simplequestions_labels")
    acc["train"] = getdata(trainp, worddic, entdic, reldic, dmp=dmp, idx=idx)
    acc["valid"] = getdata(validp, worddic, entdic, reldic, dmp=dmp, idx=idx)
    acc["test"] = getdata(testp, worddic, entdic, reldic, dmp=dmp, idx=idx)
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


def getdata(p, worddic, entdic, reldic, maxc=np.infty, dmp=False, idx=None):
    data = []
    gold = []
    cans = []
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
        if dmp:
            qcans = idx.searchallngrams(idx.getallngrams(words, topsize=5), top=10)
            cans.append(qcans)
        c += 1
        if c % 100 == 0:
            print c
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
    if dmp:     # transform goldmat and include dmp mat
        dmpmat, goldmat = makedmp(goldmat, entdic, reldic, idx)
        return datamat, goldmat, dmpmat
    else:
        return datamat, goldmat


def makedmp(goldmat, entdic, reldic, idx):
    reventdic = {v: k for k, v in entdic.items()}
    revreldic = {v: k for k, v in reldic.items()}


if __name__ == "__main__":
    argprun(run)
    #getwords("What's is plaza-midwood (wood) a type of?")