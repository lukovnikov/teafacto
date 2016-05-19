import pickle

import numpy as np, os

from theano import tensor as TT
import theano

from teafacto.core.base import Block, param, tensorops as T
from teafacto.blocks.basic import Softmax

from teafacto.blocks.rnn import SimpleSeqTransducer
from teafacto.util import argprun


def getdatamatrix(lot, maxlen, k, offset=1):
    data = np.zeros((len(lot), maxlen))
    i = 0
    while i < len(lot):
        x = lot[i][k]
        j = 0
        while j < x.shape[0]:
            data[i, j] = x[j] + offset
            j += 1
        i += 1
    return data


def tup2text(tup, word2idx, table2idx, label2idx):
    word2idxrev = {v: k for k, v in word2idx.items()}
    table2idxrev = {v: k for k, v in table2idx.items()}
    label2idxrev = {v: k for k, v in label2idx.items()}
    i = 0
    words = " ".join(map(lambda x: word2idxrev[tup[0][x]], range(len(tup[0]))))
    labels = " ".join(map(lambda x: label2idxrev[tup[2][x]], range(len(tup[0]))))
    print words
    print labels


def atiseval(preds, golds, revdic):
    """ computes accuracy, precision, recall and f-score on recognized slots"""
    assert(preds.shape[0] == golds.shape[0])
    i = 0
    tp = 0
    fp = 0
    fn = 0
    while i < preds.shape[0]:
        predslots = getslots(preds[i], revdic=revdic)
        goldslots = getslots(golds[i], revdic=revdic)
        for predslot in predslots:
            if predslot in goldslots:
                tp += 1
            else:
                fp += 1
        for goldslot in goldslots:
            if goldslot not in predslots:   # FN
                fn += 1
        i += 1
    precision = 1.0* tp / (tp + fp) if (tp + fp) > 0 else 0.
    recall = 1.0* tp / (tp + fn) if (tp + fp) > 0 else 0.
    fscore = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.
    return fscore, {"p": precision, "r": recall}, {"tp": tp, "fp": fp, "fn": fn}


def getslots(x, revdic):
    y = np.vectorize(lambda a: revdic[a] if a in revdic else "_" )(x)
    slots = []
    currentslot = None
    i = 0
    sumtingwong = False
    while i < len(y):
        ye = y[i]
        if ye == "O":   # no slot/finalize slot
            if currentslot is not None: #finalize slot
                slots.append(currentslot)
                currentslot = None
            else:   # do nothing
                pass
        elif ye[0] == "B":      # slot starting
            if currentslot is not None: #finalize slot
                slots.append(currentslot)
            else:   # do nothing
                pass
            currentslot = (ye[2:], [i]) # start new slot
        elif ye[0] == "I":      # slot continuing
            if currentslot is not None:
                if currentslot[0] == ye[2:]:
                    currentslot[1].append(i)
                else:       # wrong continuation --> finalize slot?
                    slots.append(currentslot)
                    currentslot = None
            else:   # something wrong
                print "sum ting wong"
                sumtingwong = True
        i += 1
    if sumtingwong:
        print y
    return slots


class StupidAtis(Block):
    def __init__(self, indim=100, inpembdim=50, outdim=100, **kw):
        self.E = param((indim, inpembdim), name="emb").uniform()
        self.W = param((inpembdim, outdim), name="W").uniform()
        super(StupidAtis, self).__init__(**kw)

    def apply(self, x):     # x: (batsize, seqlen)
        emb = self.E[x]     # (batsize, seqlen, inpembdim)
        outs = T.tensordot(emb, self.W, 1)  # (batsize, seqlen, outdim)
        outsf = outs.reshape((outs.shape[0] * outs.shape[1], outs.shape[2]))    # (batsize*seqlen, outdim)
        outsfsm = Softmax()(outsf)
        ret = outsfsm.reshape(outs.shape)   # (batsize, seqlen, outdim)
        return ret


class StupidAtisNative(Block):
    def __init__(self, indim=100, inpembdim=50, outdim=100, **kw):
        super(StupidAtisNative, self).__init__(**kw)
        self.E = self.add_param(param((indim, inpembdim), name="emb").uniform())
        self.W = self.add_param(param((inpembdim, outdim), name="W").uniform())

    def _apply(self, x):
        emb = self.E.d[x]
        outs = TT.tensordot(emb, self.W.d, 1)
        outsf = outs.reshape((outs.shape[0] * outs.shape[1], outs.shape[2]))
        outsfsm = TT.nnet.softmax(outsf)
        ret = outsfsm.reshape(outs.shape)
        return ret


class StupidAtisScanMod(StupidAtis):
    scanemb = False
    scanout = True
    scansm = True

    def apply(self, x):
        emb = self.recembed(x) if self.scanemb else self.E[x]     # (batsize, seqlen, inpembdim)
        outs = self.recout(emb) if self.scanout else T.tensordot(emb, self.W, 1)
        ret = self.recret(outs) if self.scansm else Softmax()(outs.reshape((outs.shape[0] * outs.shape[1], outs.shape[2]))).reshape(outs.shape)    # (batsize*seqlen, outdim)
        return ret

    def recembed(self, x):
        E = self.E
        def rec(x_t):
            return E[x_t]
        return T.scan(fn=rec, sequences=x.dimshuffle(1, 0), outputs_info=None)[0].dimshuffle(1, 0, 2)

    def recout(self, x):
        W = self.W
        def rec(x_t):
            return T.dot(x_t, W)
        return T.scan(fn=rec, sequences=x.dimshuffle(1, 0, 2), outputs_info=None)[0].dimshuffle(1, 0, 2)

    def recret(self, x):
        sm = T.nnet.softmax
        def rec(x_t):
            return sm(x_t)
        return T.scan(fn=rec, sequences=x.dimshuffle(1, 0, 2), outputs_info=None)[0].dimshuffle(1, 0, 2)


class StupidAtisScanModNative(StupidAtisNative):
    scanemb = True
    scanout = False
    scansm = False

    def _apply(self, x):
        emb = self.recembed(x) if self.scanemb else self.E.d[x]     # (batsize, seqlen, inpembdim)
        outs = self.recout(emb) if self.scanout else TT.tensordot(emb, self.W.d, 1)
        ret = self.recret(outs) if self.scansm else TT.nnet.softmax(outs.reshape((outs.shape[0] * outs.shape[1], outs.shape[2]))).reshape(outs.shape)    # (batsize*seqlen, outdim)
        return ret

    def recembed(self, x):
        E = self.E.d
        def rec(x_t):
            return E[x_t]
        return theano.scan(fn=rec, sequences=x.dimshuffle(1, 0),    outputs_info=None)[0].dimshuffle(1, 0, 2)

    def recout(self, x):
        W = self.W.d
        def rec(x_t):
            return TT.dot(x_t, W)
        return theano.scan(fn=rec, sequences=x.dimshuffle(1, 0, 2), outputs_info=None)[0].dimshuffle(1, 0, 2)

    def recret(self, x):
        sm = TT.nnet.softmax
        def rec(x_t):
            return sm(x_t)
        return theano.scan(fn=rec, sequences=x.dimshuffle(1, 0, 2), outputs_info=None)[0].dimshuffle(1, 0, 2)


class StupidAtisScan(StupidAtis):
    def apply(self, x):
        E = self.E
        W = self.W
        sm = Softmax()
        def rec(x_t):
            emb = E[x_t]
            outs = T.dot(emb, W)
            return sm(outs)
        o, _ = T.scan(fn=rec, sequences=x.dimshuffle(1, 0), outputs_info=None)
        return o.dimshuffle(1, 0, 2)


class StupidAtisScanNative(StupidAtisNative):
    def _apply(self, x):
        E = self.E.d
        W = self.W.d
        sm = TT.nnet.softmax
        def rec(x_t):
            emb = E[x_t]
            outs = TT.dot(emb, W)
            return sm(outs)
        o, _ = theano.scan(fn=rec, sequences=x.dimshuffle(1, 0, 2), outputs_info=None)
        return o.dimshuffle(1, 0, 2)




def run(p="../../../data/atis/atis.pkl", wordembdim=100, innerdim=200, lr=0.05, numbats=100, epochs=20, validinter=1, wreg=0.0003, depth=1):
    p = os.path.join(os.path.dirname(__file__), p)
    train, test, dics = pickle.load(open(p))
    word2idx = dics["words2idx"]
    table2idx = dics["tables2idx"]
    label2idx = dics["labels2idx"]
    label2idxrev = {v: k for k, v in label2idx.items()}
    train = zip(*train)
    test = zip(*test)
    print "%d training examples, %d test examples" % (len(train), len(test))
    #tup2text(train[0], word2idx, table2idx, label2idx)
    maxlen = 0
    for tup in train + test:
        maxlen = max(len(tup[0]), maxlen)

    numwords = max(word2idx.values()) + 2
    numlabels = max(label2idx.values()) + 2

    # get training data
    traindata = getdatamatrix(train, maxlen, 0).astype("int32")
    traingold = getdatamatrix(train, maxlen, 2).astype("int32")
    trainmask = (traindata > 0).astype("float32")

    # test data
    testdata = getdatamatrix(test, maxlen, 0).astype("int32")
    testgold = getdatamatrix(test, maxlen, 2).astype("int32")
    testmask = (testdata > 0).astype("float32")

    res = atiseval(testgold-1, testgold-1, label2idxrev); print res#; exit()

    # define model
    innerdim = [innerdim] * depth
    m = SimpleSeqTransducer(indim=numwords, embdim=wordembdim, innerdim=innerdim, outdim=numlabels)
    '''m = StupidAtis(inpembdim = wordembdim, indim = numwords, outdim = numlabels)
    m = StupidAtisNative(inpembdim=wordembdim, indim=numwords, outdim=numlabels)'''
    #m = StupidAtisScanMod(inpembdim=wordembdim, indim=numwords, outdim=numlabels)
    #m = StupidAtisScanModNative(inpembdim=wordembdim, indim=numwords, outdim=numlabels)

    # training
    '''m.train([traindata, trainmask], traingold).adagrad(lr=lr).grad_total_norm(5.0).seq_cross_entropy().l2(wreg)\
        .split_validate(splits=5, random=True).seq_cross_entropy().seq_accuracy().validinter(validinter)\
        .train(numbats, epochs)'''

    m.train([traindata], traingold).adagrad(lr=lr).grad_total_norm(5.0).seq_cross_entropy().l2(wreg)\
        .split_validate(splits=5, random=True).seq_cross_entropy().seq_accuracy().validinter(validinter)\
        .train(numbats, epochs)

    # predict after training
    testpredprobs = m.predict(testdata, testmask)
    testpred = np.argmax(testpredprobs, axis=2)-1
    #testpred = testpred * testmask
    #print np.vectorize(lambda x: label2idxrev[x] if x > -1 else " ")(testpred)

    evalres = atiseval(testpred, testgold-1, label2idxrev); print evalres






if __name__ == "__main__":
    argprun(run)
