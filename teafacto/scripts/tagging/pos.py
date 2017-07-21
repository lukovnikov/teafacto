from teafacto.util import argprun, ticktock, StringMatrix
from teafacto.procutil import wordids2string, wordmat2chartensor, wordmat2wordchartensor, slicer_from_flatcharseq
from IPython import embed
import numpy as np
from teafacto.core import Block, asblock

from teafacto.blocks.seq import RNNSeqEncoder
from teafacto.blocks.basic import SMO
from teafacto.blocks.word import Glove
from teafacto.blocks.word.wordrep import *
from teafacto.core.trainer import ExternalObjective
from teafacto.core.base import asblock


def loaddata(p="../../../data/pos/", rarefreq=1, task="chunk"):
    trainp = p + "train.txt"
    testp = p + "test.txt"
    traindata, traingold, wdic, tdic = loadtxt(trainp, task=task)
    testdata, testgold, wdic, tdic = loadtxt(testp, task=task, wdic=wdic, tdic=tdic)
    traindata = wordmat2wordchartensor(traindata, worddic=wdic, maskid=0)
    testdata = wordmat2wordchartensor(testdata, worddic=wdic, maskid=0)
    return (traindata, traingold), (testdata, testgold), (wdic, tdic)


def loaddata_flatcharword(p="../../../data/pos/", rarefreq=0, task="chunk"):
    trainp, testp = p + "train.txt", p + "test.txt"
    ism, osm = StringMatrix(), StringMatrix()
    ism.tokenize = lambda x: x.lower().split(" ")
    osm.tokenize = lambda x: x.split(" ")
    curdata, curgold = [], []
    i = 0
    spliti = -1
    for p in (trainp, testp):
        spliti = 0 if spliti == -1 else i
        with open(p) as f:
            for line in f:
                if len(line) < 3:
                    if len(curdata) > 0 and len(curgold) > 0:
                        ism.add(" ".join(curdata))
                        osm.add(" ".join(curgold))
                        curdata, curgold = [], []
                        i += 1
                    continue
                w, pos, chunk = line.split()
                if task == "pos":
                    t = pos
                elif task == "chunk":
                    t = chunk
                else:
                    raise Exception("unknown task for this dataset")
                curdata.append(w); curgold.append(t)
    ism.finalize(); osm.finalize()
    # get flat charseq
    ism_char = StringMatrix()
    ism_char.tokenize = lambda x: x
    for i in range(len(ism.matrix)):
        sentence = ism.pp(ism.matrix[i]) + " "
        ism_char.add(sentence)
    ism_char.finalize()
    slicer, slicermask, wordsperrow = slicer_from_flatcharseq(ism_char.matrix, wordstop=ism_char.d(" "))
    # split train test
    trainwords, testwords = ism.matrix[:spliti], ism.matrix[spliti:]
    trainchars, testchars = ism_char.matrix[:spliti], ism_char.matrix[spliti:]
    trainslice, testslice = slicer[:spliti], slicer[spliti:]
    traingold, testgold = osm.matrix[:spliti], osm.matrix[spliti:]
    return (trainwords, trainchars, trainslice, traingold),\
           (testwords, testchars, testslice, testgold),\
           (ism._dictionary, ism_char._dictionary, osm._dictionary)


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


def rebase(wordmat, srcdic, tgtdic):
    assert(srcdic["<MASK>"] == tgtdic["<MASK>"])
    assert(srcdic["<RARE>"] == tgtdic["<RARE>"])
    srctotgt = {v: tgtdic[k] if k in tgtdic else tgtdic["<RARE>"]
                for k, v in srcdic.items()}
    wordmat = np.vectorize(lambda x: srctotgt[x])(wordmat)
    return wordmat


# CHUNK EVAL
def eval_map(model, data, gold, tdic, verbose=True):
    tt = ticktock("eval", verbose=verbose)
    tt.tick("predicting")
    rtd = {v: k for k, v in tdic.items()}
    pred = model.predict(*data)
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
    if tp + fp == 0. or tp + fn == 0.:
        return -0., -0., -0.
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

    def __call__(self, *a):
        data = a[:-1]
        gold = a[-1]
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
        _, _, f1 = eval_reduce(self.tp, self.fp, self.fn)
        return f1


# POS EVAL
def tokenacceval(model, data, gold):
    pred = model.predict(*data)
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

    def __call__(self, *a):
        data = a[:-1]
        gold = a[-1]
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

    def apply(self, *x):     # (batsize, seqlen)
        enco = self.enc(*x)  # (batsize, seqlen, dim)
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
        epochs=35,
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
        mode="words",       # words or concat or gate or ctxgate or flatcharword
        gradnorm=5.,
        skiptraining=False,
        debugvalid=False,
        task="chunk",       # chunk or pos #TODO ner
    ):
    # MAKE DATA
    tt = ticktock("script")
    tt.tick("loading data")
    if mode == "flatcharword":
        (traindata, trainchars, trainslice, traingold),\
        (testdata, testchars, testslice, testgold),\
        (wdic, cdic, tdic) \
            = loaddata_flatcharword(task=task)
    else:
        (traindata, traingold), (testdata, testgold), (wdic, tdic) = loaddata(task=task)
    tt.tock("data loaded")
    g = Glove(embdim, trainfrac=embtrainfrac)
    if True:
        tt.tick("rebasing to glove dic")
        traindata = rebase(traindata, wdic, g.D)
        testdata = rebase(testdata, wdic, g.D)
        tt.tock("rebased to glove dic")
    else:
        tt.tick("doing rare")
        traindata, testdata = dorare(traindata, testdata, g, embtrainfrac=embtrainfrac, rarefreq=1)
        tt.tock("rare done")
    if inspectdata:
        revwdic = {v: k for k, v in g.D.items()}
        def pp(xs):
            return " ".join([revwdic[x] if x in revwdic else revwdic["<RARE>"]
                             for x in xs if x > 0])
        embed()

    # BUILD MODEL
    # Choice of word representation
    if mode != "words" and mode != "flatcharword":
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
    elif mode == "flatcharword":
        pass
    else:
        raise Exception("unknown mode in script")
    # tagging model
    if not mode == "flatcharword":
        enc = RNNSeqEncoder.fluent().setembedder(emb)\
            .addlayers([encdim]*layers, bidir=bidir, dropout_in=dropout).make()\
            .all_outputs()
    else:
        emb = g
        numchars = max(cdic.values()) + 1
        charenc = RNNSeqEncoder.fluent()\
            .vectorembedder(numchars, charembdim, maskid=cdic["<MASK>"])\
            .addlayers(embdim, bidir=False).make().all_outputs()
        topenc = RNNSeqEncoder.fluent().noembedder(embdim*2)\
            .addlayers([encdim]*layers, bidir=bidir, dropout_in=dropout).make()\
            .all_outputs()

        def blockfunc(wordids, charids, slicer):
            wordembs = emb(wordids)     # (batsize, wordseqlen, wordembdim)
            charencs = charenc(charids) # (batsize, charseqlen, charencdim)
            charslic = charencs[
                T.arange(slicer.shape[0]).dimshuffle(0, "x")
                    .repeat(slicer.shape[1], axis=1),
                slicer]   # (batsize, wordseqlen, charencdim)
            wordvecs = T.concatenate([wordembs, charslic], axis=2)  # (batsize, wordseqlen, wordembdim + charencdim)
            wordvecs.mask = wordembs.mask
            enco = topenc(wordvecs)     # (batsize, seqlen, encdim)
            return enco
        enc = asblock(blockfunc)

    # output tagging model
    encoutdim = encdim if not bidir else encdim * 2
    out = SMO(encoutdim, len(tdic), nobias=True)

    # final
    m = SeqTagger(enc, out)
    #charencs[np.arange(5)[:, np.newaxis].repeat(slicer.shape[1], axis=1), slicer].shape
    # TRAINING
    if mode == "words":
        traindata = traindata[:, :, 0]
        testdata = testdata[:, :, 0]
    elif mode == "concat" or mode == "gate" or mode == "ctxgate":
        tt.msg("character-level included")
    elif mode == "flatcharword":
        pass
    else:
        raise Exception("unknown mode in script")

    if task == "chunk":
        extvalid = F1Eval(m, tdic)
        evaluate_f = evaluate       # for frame-L
    elif task == "pos":
        extvalid = TokenAccEval(m)
        tokenacceval_f = tokenacceval   # for frame-L
    else:
        raise Exception("unknown task")

    if mode == "flatcharword":
        traindata = [traindata, trainchars, trainslice]
        testdata = [testdata, testchars, testslice]
    else:
        traindata = [traindata]
        testdata = [testdata]

    if not skiptraining:
        m = m.train(traindata, traingold)\
            .cross_entropy().seq_accuracy()\
            .adadelta(lr=lr).grad_total_norm(gradnorm)\
            .split_validate(splits=10)\
            .cross_entropy().seq_accuracy().extvalid(extvalid)\
            .earlystop(select=lambda x: -x[3],
                       stopcrit=7)\
            .train(numbats=numbats, epochs=epochs, _skiptrain=debugvalid)
    else:
        tt.msg("skipping training")

    if task == "chunk":
        prec, rec, f1 = evaluate(m, testdata, testgold, tdic)
        print "Precision: {} \n Recall: {} \n F-score: {}".format(prec, rec, f1)
    elif task == "pos":
        acc, num = tokenacceval(m, testdata, testgold)
        print "Token Accuracy: {}".format(1. * acc / num)


if __name__ == "__main__":
    argprun(run)

    # Initial results: 10 ep, 200D emb, 2BiGru~300D enc, lr 0.5
    # 91.32, 91.33 F1 just words
    # 92.48, 92.98, 92.59 F1 with concat
    #   92.76, 92.75 F1 with concat, 3 layers
    # 92.48, 92.25 F1 with gate
    # 92.92, 92.82, 91.52 F1 with ctxgate

    # Proper results (early stopping,...)
    # 200D emb, 2BiGru-300D enc, lr 0.5
    # 91.67@ep29 F1 just words
    # 93.34@ep51, 93.04@ep34, 92.92@ep20 F1 concat
    # 93.13@ep20, 93.42@ep43, 93.47@ep33, 92.99@ep20 F1 gate
    # 93.29@ep29, 93.19@ep27, 93.17@ep35, 93.17@ep35 F1 ctxgate