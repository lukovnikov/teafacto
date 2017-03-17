from teafacto.util import argprun, ticktock
from teafacto.procutil import wordids2string, wordmat2chartensor, wordmat2wordchartensor
from IPython import embed
import numpy as np
from teafacto.core import Block, asblock

from teafacto.blocks.seq import RNNSeqEncoder
from teafacto.blocks.basic import SMO
from teafacto.blocks.word import Glove
from teafacto.blocks.word.wordrep import *
from teafacto.core.trainer import ExternalObjective


def loaddata(p="../../../data/pos/", rarefreq=1, task="chunk"):
    trainp = p + "train.txt"
    testp = p + "test.txt"
    traindata, traingold, wdic, tdic = loadtxt(trainp, task=task)
    testdata, testgold, wdic, tdic = loadtxt(testp, task=task, wdic=wdic, tdic=tdic)
    traindata = wordmat2wordchartensor(traindata, worddic=wdic, maskid=0)
    testdata = wordmat2wordchartensor(testdata, worddic=wdic, maskid=0)
    return (traindata, traingold), (testdata, testgold), (wdic, tdic)


def loadtxt(p, wdic=None, tdic=None, task="chunk"):
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
            w, pos, chunk = line.split()
            if task == "pos":
                t = pos
            elif task == "chunk":
                t = chunk
            else:
                raise Exception("unknown task for this dataset")
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


# CHUNK EVAL
def eval_map(model, data, gold, tdic, verbose=True):
    tt = ticktock("eval", verbose=verbose)
    tt.tick("predicting")
    rtd = {v: k for k, v in tdic.items()}
    pred = model.predict(data)
    pred = np.argmax(pred, axis=2)
    mask = gold == 0
    pred[mask] = 0
    tp, fp, fn = 0., 0., 0.
    tt.tock("predicted")

    def getchunks(row):
        chunks = set()
        curstart = None
        curtag = None
        for i, e in enumerate(row):
            bio = e[0]
            tag = e[2:] if len(e) > 2 else None
            if bio == "B" or bio == "O" or \
                    (bio == "I" and tag != curtag):  # finalize current tag
                if curtag is not None:
                    chunks.add((curstart, i, curtag))
                curtag = None
                curstart = None
            if bio == "B":  # start new tag
                curstart = i
                curtag = e[2:]
        if curtag is not None:
            chunks.add((curstart, i, curtag))
        return chunks

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

    return tp, fp, fn


def eval_reduce(tp, fp, fn):
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2. * prec * rec / (prec + rec)
    return prec, rec, f1


def evaluate(model, data, gold, tdic):
    tp, fp, fn = eval_map(model, data, gold, tdic, verbose=True)
    prec, rec, f1 = eval_reduce(tp, fp, fn)
    return prec, rec, f1


class F1Eval(ExternalObjective):
    def __init__(self, model, tdic):
        super(F1Eval, self).__init__()
        self.m = model
        self.tdic = tdic
        self.tp, self.fp, self.fn = 0., 0., 0.

    def __call__(self, data, gold):
        tp, fp, fn = eval_map(self.m, data, gold, self.tdic, verbose=False)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        _, _, ret = eval_reduce(tp, fp, fn)
        return ret

    def update_agg(self, err, numex):
        pass

    def reset_agg(self):
        self.tp, self.fp, self.fn = 0., 0., 0.

    def get_agg_error(self):
        if self.tp + self.fp == 0. or self.tp + self.fn == 0.:
            return -0.
        _, _, f1 = eval_reduce(self.tp, self.fp, self.fn)
        return f1


# POS EVAL
def tokenacceval(model, data, gold):
    pred = model.predict(data)
    pred = np.argmax(pred, axis=2)
    mask = gold != 0
    corr = pred == gold
    corr *= mask
    agg = np.sum(corr)
    num = np.sum(mask)
    return agg, num


class TokenAccEval(ExternalObjective):
    def __init__(self, model):
        super(TokenAccEval, self).__init__()
        self.m = model
        self.num = 0.
        self.agg = 0.

    def __call__(self, data, gold):
        agg, num = tokenacceval(self.m, data, gold)
        self.agg += agg
        self.num += num

    def update_agg(self, err, numex):
        pass

    def reset_agg(self):
        self.num = 0.
        self.agg = 0.

    def get_agg_error(self):
        if self.num == 0.:
            return -0.
        else:
            return self.agg / self.num


class SeqTagger(Block):
    def __init__(self, enc, out, **kw):
        super(SeqTagger, self).__init__(**kw)
        self.enc = enc
        self.out = out

    def apply(self, x):     # (batsize, seqlen)
        enco = self.enc(x)  # (batsize, seqlen, dim)
        outo = self.out(enco)
        return outo


class CharEncWrap(Block):
    def __init__(self, charenc, fromdim, todim, **kw):
        super(CharEncWrap, self).__init__(**kw)
        self.enc = charenc
        self.tra = Forward(fromdim, todim, nobias=True)

    def apply(self, x):
        enco = self.enc(x)
        ret = self.tra(enco)
        return ret


def run(
        epochs=20,
        numbats=100,
        lr=0.5,
        embdim=50,
        encdim=100,
        charembdim=100,
        layers=2,
        bidir=True,
        dropout=0.3,
        embtrainfrac=1.,
        inspectdata=False,
        mode="words",       # words or concat or gate or ctxgate
        gradnorm=5.,
        skiptraining=False,
        debugvalid=False,
        task="chunk",       # chunk or pos #TODO ner
    ):
    # MAKE DATA
    tt = ticktock("script")
    tt.tick("loading data")
    (traindata, traingold), (testdata, testgold), (wdic, tdic) = loaddata(task=task)
    tt.tock("data loaded")
    g = Glove(embdim, trainfrac=embtrainfrac, worddic=wdic, maskid=0)
    tt.tick("doing rare")
    traindata, testdata = dorare(traindata, testdata, g, embtrainfrac=embtrainfrac, rarefreq=1)
    tt.tock("rare done")
    if inspectdata:
        embed()

    # BUILD MODEL
    # Choice of word representation
    if mode != "words":
        numchars = traindata[:, :, 1:].max() + 1
        charenc = RNNSeqEncoder.fluent()\
            .vectorembedder(numchars, charembdim, maskid=0)\
            .addlayers(embdim, bidir=True)\
            .make()
        charenc = CharEncWrap(charenc, embdim * 2, embdim)
    if mode == "words":
        emb = g
    elif mode == "concat":
        emb = WordEmbCharEncConcat(g, charenc)
    elif mode == "gate":
        emb = WordEmbCharEncGate(g, charenc, gatedim=embdim, dropout=dropout)
    elif mode == "ctxgate":
        gate_enc = RNNSeqEncoder.fluent()\
            .noembedder(embdim * 2)\
            .addlayers(embdim, bidir=True)\
            .add_forward_layers(embdim, activation=Sigmoid)\
            .make().all_outputs()
        emb = WordEmbCharEncCtxGate(g, charenc, gate_enc=gate_enc)
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
    elif mode == "concat" or mode == "gate" or mode == "ctxgate":
        tt.msg("character-level included")
    else:
        raise Exception("unknown mode in script")

    if task == "chunk":
        extvalid = F1Eval(m, tdic)
    elif task == "pos":
        extvalid = TokenAccEval(m)
    else:
        raise Exception("unknown task")

    if not skiptraining:
        m = m.train([traindata], traingold)\
            .cross_entropy().seq_accuracy()\
            .adadelta(lr=lr).grad_total_norm(gradnorm).exp_mov_avg(0.99)\
            .split_validate(splits=10)\
            .cross_entropy().seq_accuracy().extvalid(extvalid)\
            .takebest(f=lambda x: x[3])\
            .train(numbats=numbats, epochs=epochs, _skiptrain=debugvalid)
    else:
        tt.msg("skipping training")

    if task == "chunk":
        prec, rec, f1 = evaluate(m, testdata, testgold, tdic)
        print "Precision: {} \n Recall: {} \n F-score: {}".format(prec, rec, f1)
    elif task == "pos":
        acc, num = tokenacceval(m, testdata, testgold)
        print "Token Accuracy: {}".format(acc / num)


if __name__ == "__main__":
    argprun(run)

    # Initial results: 10 ep, 200D emb, 2BiGru~300D enc, lr 0.5
    # 91.32, 91.33 F1 just words
    # 92.48, 92.98, 92.59 F1 with concat
    #   92.76, 92.75 F1 with concat, 3 layers
    # 92.48, 92.25 F1 with gate
    # 92.92, 92.82, 91.52 F1 with ctxgate