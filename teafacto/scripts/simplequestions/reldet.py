import sys, re
from teafacto.util import argprun
from collections import OrderedDict
import numpy as np, pickle
from teafacto.blocks.seqproc import SimpleSeq2Idx, SimpleSeq2Vec, SimpleVec2Idx, MemVec2Idx, Seq2Idx


def readdata(p):
    x = pickle.load(open(p))
    def preprocessforreldet(x, numents):
        goldmat = x[1]
        return x[0], goldmat[:, 1] - numents
    entdic = x["entdic"]
    numents = x["numents"]
    newdic = {}
    for k, v in entdic.items():
        if v >= numents:
            newdic[k] = v - numents
    train = preprocessforreldet(x["train"], numents)
    valid = preprocessforreldet(x["valid"], numents)
    test  = preprocessforreldet(x["test"],  numents)
    return train, valid, test, x["worddic"], newdic


def getmemdata(reldic, worddic):
    rels = sorted(reldic.items(), key=lambda (x, y): y)
    rels = map(lambda (x, y): (filter(lambda x: len(x) > 0, re.split("[\W_]", x)), y), rels)
    allrelwords = set()
    maxlen = 0
    prevc = -1
    for rel, c in rels:
        assert(c-1 == prevc)
        prevc = c
        maxlen = max(maxlen, len(rel))
        for relw in rel:
            allrelwords.add(relw)
    relwordsnotinworddic = allrelwords.difference(set(worddic.keys()))
    for rwniw in relwordsnotinworddic:
        worddic[rwniw] = len(worddic)
    ret = [[worddic[w] for w in rel] for (rel, _) in rels]
    retmat = np.zeros((len(rels), maxlen)).astype("int32") - 1
    i = 0
    for r in ret:
        retmat[i, :len(r)] = r
        i += 1
    return retmat


def evaluate(pred, gold):
    return np.sum(gold == pred) * 100. / gold.shape[0]


def run(
        epochs=10,
        numbats=100,
        numsam=10000,
        lr=0.1,
        datap="../../../data/simplequestions/datamat.word.pkl",
        embdim=100,
        innerdim=200,
        wreg=0.00005,
        bidir=False,
        keepmincount=5,
        mem=False,
        sameenc=False,
        ):

    (traindata, traingold), (validdata, validgold), (testdata, testgold), worddic, entdic\
        = readdata(datap)

    if mem:
        memdata = getmemdata(entdic, worddic)

    print traindata.shape, testdata.shape

    numwords = max(worddic.values()) + 1
    numrels = max(entdic.values()) + 1
    print numwords, numrels

    if bidir:
        encinnerdim = innerdim/2
    else:
        encinnerdim = innerdim

    enc = SimpleSeq2Vec(indim=numwords, inpembdim=embdim, innerdim=encinnerdim, maskid=-1, bidir=bidir)

    if mem:
        memenc = enc if sameenc else SimpleSeq2Vec(indim=numwords, inpembdim=embdim, innerdim=innerdim, maskid=-1)
        dec = MemVec2Idx(memenc, memdata, memdim=innerdim)
    else:
        dec = SimpleVec2Idx(indim=innerdim, outdim=numrels)

    m = Seq2Idx(enc, dec)

    m = m.train([traindata], traingold).adagrad(lr=lr).l2(wreg).grad_total_norm(1.0).cross_entropy()\
        .validate_on([validdata], validgold).accuracy().cross_entropy().takebest()\
        .train(numbats=numbats, epochs=epochs)

    pred = m.predict(testdata)
    print pred.shape
    evalres = evaluate(np.argmax(pred, axis=1), testgold)
    print str(evalres) + "%"


if __name__ == "__main__":
    argprun(run)