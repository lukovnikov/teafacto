import numpy as np
import theano
from theano import tensor as T

from teafacto.core.rnn import RNNMask, GRU, RNNEncoder
from teafacto.core.trainutil import SMBase, Predictor, Saveable
from teafacto.lm import Glove
from IPython import embed


class MultiQABaseSM(SMBase, Predictor, Saveable):

    def defmodel(self): # model behaves correctly
        golds = T.ivector("answers")
        iq, ia, ib, ic, id = T.imatrices("q", "a", "b", "c", "d") # (batsize, seqlen)
        q, a, b, c, d = [self.wemb[x, :] for x in [iq, ia, ib, ic, id]] # (batsize, seqlen, dim)
        # sum
        qenc = self.encodeQ(q)
        aenc, benc, cenc, denc = map(self.encodeA, [a,b,c,d])
        dots = [self.encScore(qenc, x) for x in [aenc, benc, cenc, denc]]
        #dots = [T.sqr((qenc - x).norm(2, axis=1)).reshape((x.shape[0], 1)) for x in [aenc, benc, cenc, denc]]
        dots = T.concatenate(dots, axis=1) # (batsize, 4)
        probs = T.nnet.softmax(dots)
        #embed()
        return probs, golds, [iq, ia, ib, ic, id, golds]

    def getpredictfunction(self):
        probs, _, inps = self.defmodel()
        scoref = theano.function(inputs=inps[:5], outputs=probs)
        def pref(data):
            args = [np.asarray(i).astype("int32") for i in [data[:, 0, :], data[:, 1, :], data[:, 2, :], data[:, 3, :], data[:, 4, :]]]
            outprobs = scoref(*args)    # (batsize, 4)
            return outprobs.argmax(axis=1) # (batsize,)
        return pref

    def getsamplegen(self, data, labels, onebatch=False): # data: ? list of (batsize, seqlen), seqlen for Q is different than for A's
        if onebatch:                                      # works correctly (DONE: shapes inspected)
            batsize = data.shape[0]
        else:
            batsize = self.batsize
        sampleoffsett = [0]
        idxs = np.arange(0, data.shape[0], 1, dtype="int32")
        np.random.shuffle(idxs)
        def samplegen():
            start = sampleoffsett[0]
            end = min(sampleoffsett[0] + batsize, idxs.shape[0])
            selidxs = idxs[start:end]
            datasample = data[selidxs, :].astype("int32")
            labelsample = labels[selidxs].astype("int32")
            sampleoffsett[0] += end-start if end < idxs.shape[0] else 0
            #embed()
            return datasample[:, 0, :], \
                   datasample[:, 1, :], \
                   datasample[:, 2, :], \
                   datasample[:, 3, :], \
                   datasample[:, 4, :], \
                   labelsample
        return samplegen


class DotSumEncSM(MultiQABaseSM):
    def __init__(self, dim=50, wembs=None, **kw):
        super(DotSumEncSM, self).__init__(**kw)
        self.dim = dim
        if wembs is None:
            wembs = Glove(dim)
        self.wemb = wembs.theano

    def encodeQ(self, qvar): # (batsize, seqlen, dim) ==> (batsize, dim)
        sum = T.sum(qvar, axis=1)
        eps = 10e-10 # to avoid division by zero
        sum = (sum.T / (sum.norm(2, axis=1) + eps)).T
        return sum

    def encodeA(self, avar):
        return self.encodeQ(avar)

    def encScore(self, qenc, aenc):
        return T.batched_dot(qenc, aenc).reshape((aenc.shape[0], 1))

    @property
    def depparameters(self):
        return set()

    @property
    def ownparameters(self):
        return {self.wemb}


class RNNWeightedSumEncDotSM(MultiQABaseSM):
    def __init__(self, dim=50, wembs=None, **kw):
        super(RNNWeightedSumEncDotSM, self).__init__(**kw)
        self.dim = dim
        self.innerdim = dim
        self.mask = RNNMask() + GRU(dim=self.dim, innerdim=self.innerdim, wreg=self.wreg) # one mask for Q's and A's
        if wembs is None:
            wembs = Glove(dim)
        self.wemb = wembs.theano


class RNNMaskedSumEncDotSM(MultiQABaseSM):
    def __init__(self, dim=50, wembs=None, **kw):
        super(RNNMaskedSumEncDotSM, self).__init__(**kw)
        self.dim = dim
        self.innerdim = dim
        self.mask = RNNMask() + GRU(dim=self.dim, innerdim=self.innerdim, wreg=self.wreg) # one mask for Q's and A's
        if wembs is None:
            wembs = Glove(dim)
        self.wemb = wembs.theano

    def encodeQ(self, qvar):
        qmasked = self.mask.mask(qvar)
        sum = T.sum(qmasked, axis=1)
        eps = 10e-10 # to avoid division by zero
        sum = (sum.T / (sum.norm(2, axis=1) + eps)).T
        return sum

    def encodeA(self, avar):
        sum = T.sum(avar, axis=1)
        eps = 10e-10 # to avoid division by zero
        sum = (sum.T / (sum.norm(2, axis=1) + eps)).T
        return sum

    def encScore(self, qenc, aenc):
        return T.batched_dot(qenc, aenc).reshape((aenc.shape[0], 1))

    @property
    def depparameters(self):
        return self.mask.parameters

    @property
    def ownparameters(self):
        return set()


class QAEncDotSM(MultiQABaseSM):
    def __init__(self, dim=50, innerdim=100, wembs=None, **kw):
        super(QAEncDotSM, self).__init__(**kw)
        self.dim = dim
        self.innerdim = innerdim
        self.qencoder = RNNEncoder() + GRU(dim=self.dim, innerdim=self.innerdim, wreg=self.wreg)
        self.aencoder = RNNEncoder() + GRU(dim=self.dim, innerdim=self.innerdim, wreg=self.wreg)
        if wembs is None:
            wembs = Glove(dim)
        self.wemb = wembs.theano

    def encodeQ(self, qvar):
        return self.qencoder.encode(qvar)

    def encodeA(self, avar):
        return self.aencoder.encode(avar)

    def encScore(self, qenc, aenc):
        return T.batched_dot(qenc, aenc).reshape((aenc.shape[0], 1))

    @property
    def depparameters(self):
        return self.qencoder.parameters.union(self.aencoder.parameters)

    @property
    def ownparameters(self):
        return set() #{self.wemb}