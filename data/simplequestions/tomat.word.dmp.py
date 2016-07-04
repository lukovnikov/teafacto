import pickle, sys
from IPython import embed
import numpy as np

from teafacto.datahelp.labelsearch import SimpleQuestionsLabelIndex
from teafacto.util import argprun, tokenize


def loadlabels(labelp="labels.map"):
    """ Loads labels from file (entity labels) """
    ret = {}
    for line in open(labelp):
        ns = line[:-1].split("\t")
        ret[ns[0]] = getwords(ns[1])
    print "labels loaded"
    return ret


def getallwords(labeldic, *dataps):
    """ Builds global worddic, updates labeldic with rel names """
    worddic = {"<RARE>": 0}
    lworddic = {}
    for datap in dataps:
        with open(datap) as dataf:
            for line in dataf:
                q, a = (line[:-1] if line[-1] == "\n" else line).split("\t")
                s, p = a.split()
                words = getwords(q)     # words in the question sentence
                relwords = getwords(p)
                labeldic[p] = relwords
                entwords = labeldic[s] if s in labeldic else []
                relwords.extend(entwords)
                for word in words:
                    if word not in worddic:
                        worddic[word] = len(worddic)
                for word in relwords:
                    if word not in lworddic and word not in worddic:
                        lworddic[word] = len(lworddic)
    numwordsinq = len(worddic)
    worddic.update({k: v + numwordsinq for k, v in lworddic.items()})
    print len(worddic)
    return worddic, labeldic, numwordsinq


def run(trainp="fb_train.tsv",
        testp="fb_test.tsv",
        validp="fb_valid.tsv",
        outp="datamat.word.dmp.pkl",
        labelp="labels.map",
        host="localhost",
        dmp=True):
    entdic = {}
    reldic = {}
    acc = {}
    labeldic = loadlabels(labelp)
    worddic, labeldic, qdicsize \
        = getallwords(labeldic, trainp, validp, testp)
    idx = None
    if dmp:
        idx = SimpleQuestionsLabelIndex(host=host, index="simplequestions_labels")
    #embed()
    acc["train"] = getdata(trainp, worddic, entdic, reldic, dmp=dmp, idx=idx)
    acc["valid"] = getdata(validp, worddic, entdic, reldic, dmp=dmp, idx=idx)
    acc["test"] = getdata(testp, worddic, entdic, reldic, dmp=dmp, idx=idx)
    acc["worddic"] = worddic
    acc["numqwords"] = qdicsize
    numents = len(entdic)
    acc["train"][1][:, 1] += numents
    acc["valid"][1][:, 1] += numents
    acc["test"][1][:, 1] += numents
    reldic = {k: v+numents for k, v in reldic.items()}
    entdic.update(reldic)
    print len(entdic)
    acc["entdic"] = entdic
    acc["numents"] = numents
    acc["labels"] = {entdic[k] if k in entdic else k:
                     [worddic[ve] if ve in worddic else worddic["<RARE>"] for ve in v]
                     for k, v in labeldic.items()}
    pickle.dump(acc, open(outp, "w"))


def getwords(s):
    return tokenize(s)


def getdata(p, worddic, entdic, reldic, maxc=np.infty, dmp=False, idx=None, maxcan=1000):
    # builds index matrices out of data, updates entdic, reldic
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
        if s not in entdic:
            entdic[s] = len(entdic)
        if p not in reldic:
            reldic[p] = len(reldic)
        wordidx = map(lambda x: worddic[x] if x in worddic else worddic["<RARE>"], words)
        data.append(wordidx)
        gold.append([entdic[s], reldic[p]])
        if dmp:
            qcans = idx.searchallngrams(idx.getallngrams(words, topsize=None), top=20)
            qcans = qcans.keys()
            for qcan in qcans:
                if qcan not in entdic:
                    entdic[qcan] = len(entdic)
            cans.append(map(lambda x: entdic[x], qcans))
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
        return datamat, goldmat, cans
    else:
        return datamat, goldmat


if __name__ == "__main__":
    argprun(run)
    #getwords("What's is plaza-midwood (wood) a type of?")