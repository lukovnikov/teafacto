import pickle

import numpy as np
from pympler.asizeof import asizeof
from pympler.tracker import SummaryTracker

from teafacto.search import SeqTransDecSearch
from teafacto.blocks.seqproc import SimpleSeqTransDec
from teafacto.scripts.atis.atisseqtrans import getdatamatrix, atiseval
from teafacto.util import argprun


def shiftdata(x, right=1):
    if isinstance(x, np.ndarray):
        return np.concatenate([np.zeros_like(x[:, 0:right]), x[:, :-right]], axis=1)
    else:
        raise Exception("can not shift this")


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
    testpred, _ = s.decode(testdata)
    testpred = testpred * testmask

    evalres = atiseval(testpred-1, testgold-1, label2idxrev); print evalres

    #testpredprobs = m.predict(testdata, shiftdata(testgold), testmask)
    #testpred = np.argmax(testpredprobs, axis=2)-1
    #testpred = testpred * testmask
    #print np.vectorize(lambda x: label2idxrev[x] if x > -1 else " ")(testpred)



if __name__ == "__main__":
    argprun(run, epochs=10)
