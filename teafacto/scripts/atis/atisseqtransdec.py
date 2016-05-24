import pickle

import numpy as np
from pympler.tracker import SummaryTracker
from pympler.asizeof import asizeof

from teafacto.blocks.rnn import SimpleSeqTransducer, SimpleSeqTransDec
from teafacto.util import argprun
from teafacto.scripts.atis.atisseqtrans import getdatamatrix, atiseval
from teafacto.users.modelusers import RecPredictor


def shiftdata(x, right=1):
    if isinstance(x, np.ndarray):
        return np.concatenate([np.zeros_like(x[:, 0:right]), x[:, :-right]], axis=1)
    else:
        raise Exception("can not shift this")


class Searcher(object):
    def __init__(self, model, beamsize=1, **kw):
        super(Searcher, self).__init__(**kw)
        self.beamsize = beamsize
        self.model = model
        self.mu = RecPredictor(model)


class SeqTransDecSearch(Searcher):
    # responsible for generating recappl prediction function from recappl of decoder
    """ Default: greedy search strategy """
    def decode(self, inpseq):
        stop = False
        i = 0
        curout = np.zeros((inpseq.shape[0])).astype("int32")
        accprobs = np.ones((inpseq.shape[0]))
        outs = []
        while not stop:
            curinp = inpseq[:, i]
            curprobs = self.mu.feed(curinp, curout)
            accprobs *= np.max(curprobs, axis=1)
            curout = np.argmax(curprobs, axis=1).astype("int32")
            outs.append(curout)
            i += 1
            stop = i == inpseq.shape[1]
        #print accprobs
        return np.stack(outs).T     # TODO: check with previous impl

    def decode2(self, inpseq):       # inpseq: idx^(batsize, seqlen)
        i = 0
        stop = False
        # prevpreds = [np.zeros((inpseq.shape[0], 1))]*self.beamsize
        acc = np.zeros((inpseq.shape[0], 1)).astype("int32")
        accprobs = np.ones((inpseq.shape[0]))
        while not stop:
            curinpseq = inpseq[:, :i+1]
            print curinpseq.shape
            curprobs = self.model.predict(curinpseq, acc)   # curpred: f32^(batsize, prevpred.seqlen, numlabels)
            curpreds = np.argmax(curprobs, axis=2).astype("int32")
            accprobs = np.max(curprobs, axis=2)[:, -1] * accprobs
            acc = np.concatenate([acc, curpreds[:, -1:]], axis=1)
            i += 1
            stop = i == inpseq.shape[1]
        ret = acc[:, 1:]
        finalprobs = np.max(curprobs, axis=2).prod(axis=1)
        print np.linalg.norm(finalprobs - accprobs), np.allclose(finalprobs, accprobs)
        assert(ret.shape == inpseq.shape)
        return ret


def run(p="../../../data/atis/atis.pkl", wordembdim=70, lablembdim=70, innerdim=300, lr=0.05, numbats=100, epochs=20, validinter=1, wreg=0.0003, depth=1):
    tracker = SummaryTracker()
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

    print asizeof(traindata)

    # define model
    innerdim = [innerdim] * depth
    m = SimpleSeqTransDec(indim=numwords, inpembdim=wordembdim, outembdim=lablembdim, innerdim=innerdim, outdim=numlabels)

    # training
    m = m.train([traindata, shiftdata(traingold), trainmask], traingold).adagrad(lr=lr).grad_total_norm(5.0).seq_cross_entropy().l2(wreg)\
        .split_validate(splits=5, random=True).seq_cross_entropy().seq_accuracy().validinter(validinter).takebest()\
        .train(numbats, epochs)

    # predict after training
    s = SeqTransDecSearch(m)
    testpred = s.decode(testdata)
    testpred = testpred * testmask

    evalres = atiseval(testpred-1, testgold-1, label2idxrev); print evalres

    #testpredprobs = m.predict(testdata, shiftdata(testgold), testmask)
    #testpred = np.argmax(testpredprobs, axis=2)-1
    #testpred = testpred * testmask
    #print np.vectorize(lambda x: label2idxrev[x] if x > -1 else " ")(testpred)




if __name__ == "__main__":
    argprun(run, epochs=1)
