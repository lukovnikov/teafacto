import pickle

import numpy as np

from teafacto.core.base import Block
from teafacto.blocks.basic import VectorEmbed, Softmax
from teafacto.blocks.rnn import SimpleSeqTransducer, SimpleSeqTransDec, RecurrentStack, BiRNU, SeqDecoder, SeqTransducer
from teafacto.blocks.attention import Attention, WeightedSumAttCon, DotprodAttGen, LinearGateAttentionGenerator
from teafacto.blocks.rnu import GRU
from teafacto.util import argprun, issequence
from teafacto.scripts.atis.atisseqtrans import getdatamatrix, atiseval


class SeqEncDecAtt(Block):
    def __init__(self,
                 inpvocsize=400,
                 inpembdim=50,
                 outvocsize=100,
                 outembdim=50,
                 encdim=100,
                 decdim=100,
                 attdim=100,
                 bidir=False,
                 rnu=GRU,
                 outconcat=True,
                 inconcat=True, **kw):
        super(SeqEncDecAtt, self).__init__(**kw)
        self.inpvocsize = inpvocsize
        self.outvocsize = outvocsize
        self.inpembdim = inpembdim
        self.outembdim = outembdim
        self.encinnerdim = [encdim] if not issequence(encdim) else encdim
        self.decinnerdim = [decdim] if not issequence(decdim) else decdim
        self.attdim = attdim
        self.rnu = rnu
        self.bidir = bidir

        # encoder stack
        self.inpemb = VectorEmbed(indim=self.inpvocsize, dim=self.inpembdim)
        self.encrnus = []
        dims = [self.inpembdim] + self.encinnerdim
        i = 1
        while i < len(dims):
            if self.bidir:
                rnu = BiRNU.fromrnu(self.rnu, dim=dims[i-1], innerdim=dims[i])
            else:
                rnu = self.rnu(dim=dims[i-1], innerdim=dims[i])
            self.encrnus.append(rnu)
            i += 1
        self.encoder = RecurrentStack(*([self.inpemb] + self.encrnus))

        # attention
        self.attgen = LinearGateAttentionGenerator(indim=self.encinnerdim[-1] + self.decinnerdim[-1], attdim=self.attdim)
        self.attcon = WeightedSumAttCon()

        # decoder
        self.outemb = VectorEmbed(indim=self.outvocsize, dim=self.outembdim)
        self.decrnus = []
        dims = [self.outembdim + self.encinnerdim[-1]] + self.decinnerdim
        i = 1
        while i < len(dims):
            self.decrnus.append(self.rnu(dim=dims[i-1], innerdim=dims[i]))
            i += 1
        self.decoder = SeqDecoder(
            [self.outemb] + self.decrnus,
            attention=Attention(self.attgen, self.attcon),
            innerdim=self.encinnerdim[-1] + self.decinnerdim[-1],
            outconcat=outconcat,
            inconcat=inconcat,
        )

    def apply(self, inpseq, outseq, maskseq=None):
        enco = self.encoder(inpseq)
        deco = self.decoder(enco, outseq)
        ret = SeqTransducer.applymask(deco, maskseq)
        return ret



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


class SeqEncDecAttSearch(Searcher):
    """ Default: greedy search strategy """
    def decode(self, inpseq):       # inpseq: idx^(batsize, seqlen)
        i = 0
        stop = False
        # prevpreds = [np.zeros((inpseq.shape[0], 1))]*self.beamsize
        acc = np.zeros((inpseq.shape[0], 1)).astype("int32")
        accprobs = np.ones((inpseq.shape[0]))
        while not stop:
            curprobs = self.model.predict(inpseq, acc)   # curpred: f32^(batsize, prevpred.seqlen, numlabels)
            curpreds = np.argmax(curprobs, axis=2).astype("int32")
            accprobs = np.max(curprobs, axis=2)[:, -1] * accprobs
            acc = np.concatenate([acc, curpreds[:, -1:]], axis=1)
            i += 1
            stop = i == inpseq.shape[1]
        ret = acc[:, 1:]
        finalprobs = np.max(curprobs, axis=2).prod(axis=1)
        print np.linalg.norm(finalprobs - accprobs)
        assert(ret.shape == inpseq.shape)
        return ret



def run(p="../../../data/atis/atis.pkl", wordembdim=70, lablembdim=70, innerdim=300, lr=0.05, numbats=100, epochs=20, validinter=1, wreg=0.0003, depth=1, attdim=300):
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
    m = SeqEncDecAtt(
        inpvocsize=numwords,
        inpembdim=wordembdim,
        outvocsize=numlabels,
        outembdim=lablembdim,
        encdim=innerdim,
        decdim=innerdim,
        attdim=attdim,
    )

    # training
    m.train([traindata, shiftdata(traingold), trainmask], traingold).adagrad(lr=lr).grad_total_norm(5.0).seq_cross_entropy().l2(wreg)\
        .validate_on([testdata, shiftdata(testgold), testmask], testgold).seq_cross_entropy().seq_accuracy().validinter(validinter)\
        .train(numbats, epochs)

    # predict after training
    s = SeqEncDecAttSearch(m)
    testpred = s.decode(testdata)
    testpred = testpred * testmask
    #testpredprobs = m.predict(testdata, shiftdata(testgold), testmask)
    #testpred = np.argmax(testpredprobs, axis=2)-1
    #testpred = testpred * testmask
    #print np.vectorize(lambda x: label2idxrev[x] if x > -1 else " ")(testpred)

    evalres = atiseval(testpred-1, testgold-1, label2idxrev); print evalres


if __name__ == "__main__":
    argprun(run, epochs=0)
