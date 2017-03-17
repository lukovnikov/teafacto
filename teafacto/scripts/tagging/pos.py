from teafacto.util import argprun, ticktock
from teafacto.procutil import wordids2string, wordmat2chartensor, wordmat2wordchartensor
from IPython import embed
import numpy as np
from teafacto.core import Block, asblock

from teafacto.blocks.seq import RNNSeqEncoder
from teafacto.blocks.basic import SMO
from teafacto.blocks.word import Glove
from teafacto.blocks.word.wordrep import *


def loaddata(p="../../../data/pos/", rarefreq=1):
    trainp = p + "train.txt"
    testp = p + "test.txt"
    traindata, traingold, wdic, tdic = loadtxt(trainp)
    testdata, testgold, wdic, tdic = loadtxt(testp, wdic=wdic, tdic=tdic)
    traindata = wordmat2wordchartensor(traindata, worddic=wdic, maskid=0)
    testdata = wordmat2wordchartensor(testdata, worddic=wdic, maskid=0)
    return (traindata, traingold), (testdata, testgold), (wdic, tdic)


def loadtxt(p, wdic=None, tdic=None):
    wdic = {"<MASK>": 0, "<RARE>": 1} if wdic is None else wdic
    tdic = {"<MASK>": 0} if tdic is None else tdic
    data, gold = [], []
    maxlen = 0
    curdata = []
    curgold = []
    with open(p) as f:
        for line in f:
            if len(line) < 3:
                if len(curdata) > 0 and len(curgold) > 0:
                    data.append(curdata)
                    gold.append(curgold)
                    maxlen = max(maxlen, len(curdata))
                    curdata = []
                    curgold = []
                continue
            w, pos, t = line.split()
            w = w.lower()
            if w not in wdic:
                wdic[w] = len(wdic)
            if t not in tdic:
                tdic[t] = len(tdic)
            curdata.append(wdic[w])
            curgold.append(tdic[t])
    datamat = np.zeros((len(data), maxlen), dtype="int32")
    goldmat = np.zeros((len(data), maxlen), dtype="int32")
    for i in range(len(data)):
        datamat[i, :len(data[i])] = data[i]
        goldmat[i, :len(gold[i])] = gold[i]
    return datamat, goldmat, wdic, tdic


def dorare(traindata, testdata, glove, rarefreq=1, embtrainfrac=0.0):
    counts = np.unique(traindata, return_counts=True)
    rarewords = set(counts[0][counts[1] <= rarefreq])
    goodwords = set(counts[0][counts[1] > rarefreq])
    traindata[:, :, 0] = np.vectorize(lambda x: glove.D["<RARE>"] if x in rarewords else x)(traindata[:, :, 0])
    if embtrainfrac == 0.0:
        goodwords = goodwords.union(glove.allwords)
    testdata[:, :, 0] = np.vectorize(lambda x: glove.D["<RARE>"] if x not in goodwords else x)(testdata[:, :, 0])
    return traindata, testdata


def evaluate(model, data, gold, tdic):
    rtd = {v: k for k, v in tdic.items()}
    pred = model.predict(data)
    pred = np.argmax(pred, axis=2)
    mask = gold == 0
    pred[mask] = 0
    tp, fp, fn = 0., 0., 0.

    def getchunks(row):
        chunks = set()
        curstart = None
        curtag = None
        for i, e in enumerate(row):
            bio = e[0]
            tag = e[2:] if len(e) > 2 else None
            if bio == "B" or bio == "O" or \
                    (bio == "I" and tag != curtag):     # finalize current tag
                if curtag is not None:
                    chunks.add((curstart, i, curtag))
                curtag = None
                curstart = None
            if bio == "B":                    # start new tag
                curstart = i
                curtag = e[2:]
        if curtag is not None:
            chunks.add((curstart, i, curtag))
        return chunks

    tt = ticktock("eval")
    tt.tick("evaluating")

    for i in range(len(gold)):
        goldrow = [rtd[x] for x in list(gold[i]) if x > 0]
        predrow = [rtd[x] for x in list(pred[i]) if x > 0]
        goldchunks = getchunks(goldrow)
        predchunks = getchunks(predrow)
        tpp = goldchunks.intersection(predchunks)
        fpp = predchunks.difference(goldchunks)
        fnn = goldchunks.difference(predchunks)
        tp += len(tpp)
        fp += len(fpp)
        fn += len(fnn)
        tt.progress(i, len(gold), live=True)

    tt.tock("evaluated")

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2. * prec * rec / (prec + rec)
    return prec, rec, f1


class SeqTagger(Block):
    def __init__(self, enc, out, **kw):
        super(SeqTagger, self).__init__(**kw)
        self.enc = enc
        self.out = out

    def apply(self, x):     # (batsize, seqlen)
        enco = self.enc(x)  # (batsize, seqlen, dim)
        outo = self.out(enco)
        return outo


def run(
        epochs=10,
        numbats=100,
        lr=0.5,
        embdim=50,
        encdim=100,
        layers=2,
        bidir=True,
        dropout=0.3,
        embtrainfrac=1.,
        inspectdata=False,
        mode="words",
        gradnorm=5.,
        skiptraining=False,
    ):
    # MAKE DATA
    tt = ticktock("script")
    tt.tick("loading data")
    (traindata, traingold), (testdata, testgold), (wdic, tdic) = loaddata()
    tt.tock("data loaded")
    g = Glove(embdim, trainfrac=embtrainfrac, worddic=wdic, maskid=0)
    tt.tick("doing rare")
    traindata, testdata = dorare(traindata, testdata, g, embtrainfrac=embtrainfrac, rarefreq=1)
    tt.tock("rare done")
    if inspectdata:
        embed()

    # BUILD MODEL
    # Choice of word representation
    if mode == "words":
        emb = g
    else:
        raise Exception("unknown mode in script")
    # tagging model
    enc = RNNSeqEncoder.fluent().setembedder(emb)\
        .addlayers([encdim]*layers, bidir=bidir, dropout_in=dropout).make()\
        .all_outputs()

    # output tagging model
    encoutdim = encdim if not bidir else encdim * 2
    out = SMO(encoutdim, len(tdic), nobias=True)

    # final
    m = SeqTagger(enc, out)

    # TRAINING
    if mode == "words":
        traindata = traindata[:, :, 0]
        testdata = testdata[:, :, 0]
    else:
        raise Exception("unknown mode in script")

    if not skiptraining:
        m.train([traindata], traingold)\
            .cross_entropy().seq_accuracy()\
            .adadelta(lr=lr).grad_total_norm(gradnorm)\
            .validate_on([testdata], testgold)\
            .cross_entropy().seq_accuracy()\
            .train(numbats=numbats, epochs=epochs)
    else:
        tt.msg("skipping training")

    prec, rec, f1 = evaluate(m, testdata, testgold, tdic)

    print prec, rec, f1


if __name__ == "__main__":
    argprun(run)