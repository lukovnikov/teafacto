from theano.tensor.extra_ops import to_one_hot

__author__ = 'denis'
import theano
from theano import tensor as T
import numpy as np, math
from tf import TFSGD
from utils import *
from math import ceil, floor
from datetime import datetime as dt
from IPython import embed
import sys, os, cPickle as pickle
from rnnkm import Saveable, Profileable, Predictor
from rnn import GRU, LSTM

class SGDBase(object):
    def __init__(self, maxiter=50, lr=0.0001, numbats=100, wreg=0.00001, **kw):
        self.maxiter = maxiter
        self.lr = lr
        self.numbats = numbats
        self.wreg = wreg
        self.tlr = theano.shared(np.float32(self.lr), name="lr")
        self.tnumbats = theano.shared(np.float32(self.numbats), name="numbats")
        self.twreg = theano.shared(np.float32(self.wreg), name="wreg")
        super(SGDBase, self).__init__(**kw)

    def gettrainf(self, finps, fouts, cost):
        params = self.ownparams + self.depparams
        grads = T.grad(cost, wrt=params)
        updates = self.getupdates(params, grads)
        showgraph(updates[0][1])
        return theano.function(inputs=finps,
                               outputs=fouts,
                               updates=updates,
                               profile=self._profiletheano)

    def getupdates(self, params, grads):
        updates = []
        for (p, g) in zip(params, grads):
            update = (p, (p - self.tlr * self.tnumbats * g).astype("float32"))
            updates.append(update)
        return updates

    def getnormf(self):
        return None

    def trainloop(self, trainf, validf=None, evalinter=1, normf=None, average_err=True):
        err = []
        stop = False
        itercount = 1
        evalcount = evalinter
        #if normf:
        #    normf()
        while not stop:
            print("iter %d/%d" % (itercount, self.maxiter))
            start = dt.now()
            erre = trainf()
            if average_err:
                erre /= self.numbats
            if normf:
                normf()
            if itercount == self.maxiter:
                stop = True
            itercount += 1
            err.append(erre)
            print(erre)
            print("iter done in %f seconds" % (dt.now() - start).total_seconds())
            evalcount += 1
            if self._autosave:
                self.save()
        return err

    def getbatchloop(self, trainf, samplegen):
        '''
        returns the batch loop, loaded with the provided trainf training function and samplegen sample generator
        '''
        numbats = self.numbats

        def batchloop():
            c = 0
            prevperc = -1.
            maxc = numbats
            terr = 0.
            while c < maxc:
                #region Percentage counting
                perc = round(c*100./maxc)
                if perc > prevperc:
                    sys.stdout.write("iter progress %.0f" % perc + "% \r")
                    sys.stdout.flush()
                    prevperc = perc
                #endregion
                sampleinps = samplegen()
                terr += trainf(*sampleinps)[0]
                c += 1
            return terr
        return batchloop


class XRNNKMSGDSM(SGDBase, Saveable, Profileable, Predictor):
    def __init__(self, dim=20, innerdim=20, vocabsize=1000, **kw):
        self.dim = dim
        self.innerdim = innerdim
        self.W = None
        self.sm = None
        self.vocabsize = vocabsize
        self.initvars()
        super(XRNNKMSGDSM, self).__init__(**kw)

    def initvars(self):
        offset = 0.5
        scale = 0.1
        self.W = theano.shared((np.random.random((self.vocabsize, self.dim)).astype("float32")-offset)*scale, name="W")
        self.sm = theano.shared((np.random.random((self.dim, self.vocabsize)).astype("float32")-offset)*scale, name="sm")

    def train(self, trainX, labels, evalinter=10): # X: z, x, y, v OR r, s, o, v
        self.batsize = int(ceil(trainX.shape[0]*1./self.numbats))
        self.tbatsize = theano.shared(np.int32(self.batsize))
        probs, inps, gold = self.defmodel()
        tErr = self.geterr(probs, gold)
        tReg = self.getreg()
        tCost = tErr + tReg
        showgraph(tCost)
        #embed() # tErr.eval({inps[0]: [0], inps[1]:[10], gold: [1]})

        trainf = self.gettrainf(inps+[gold], [tErr, tCost], tCost)
        err = self.trainloop(trainf=self.getbatchloop(trainf, self.getsamplegen(trainX, labels)),
                             evalinter=evalinter,
                             normf=self.getnormf())
        return err

    def defmodel(self):
        xidx, yidx, zidx = T.ivectors("xidx", "yidx", "zidx")
        xemb, yemb = self.embed(xidx, yidx)
        predemb = xemb + yemb
        preds = T.dot(predemb, self.sm)
        probs = T.nnet.softmax(preds)
        return probs, [xidx, yidx], zidx

    def embed(self, *idxs):
        return tuple(map(lambda x: self.W[x, :], idxs))

    def geterr(self, probs, gold): # probs: (batsize, vocabsize)-float, gold: (batsize,)-int
        return -T.mean(T.log(probs[T.arange(self.tbatsize), gold])) # cross-entropy
        #return T.sum(1-probs[:, gold]) theano.tensor.

    def getreg(self, regf=lambda x: T.sum(x**2), factor=1./2):
        return factor * reduce(lambda x, y: x + y,
                               map(lambda x: regf(x) * self.twreg,
                                   self.ownparams))

    @property
    def ownparams(self):
        return [self.W, self.sm]
        #return [self.W]

    @property
    def depparams(self):
        return []

    def getsamplegen(self, trainX, labels):
        batsize = self.batsize

        def samplegen():
            nonzeroidx = sorted(np.random.randint(0, trainX.shape[0], size=(batsize,)).astype("int32"))
            trainXsample = trainX[nonzeroidx, :].astype("int32")
            labelsample = labels[nonzeroidx].astype("int32")
            return [trainXsample[:, 0], trainXsample[:, 1], labelsample]     # [[s*], [r*], [o*]]
        return samplegen

    def getpredictfunction(self):
        probs, inps, gold = self.defmodel()
        score = probs[:, gold]
        scoref = theano.function(inputs=inps+[gold], outputs=score)
        def pref(s, r, o):
            args = [np.asarray(i).reshape((1,)).astype("int32") for i in [s, r, o]]
            return scoref(*args)
        return pref


class GRUKMSGDSM(XRNNKMSGDSM):
    def __init__(self, **kw):
        super(GRUKMSGDSM, self).__init__(**kw)
        self.rnnu = GRU(dim=self.dim, wreg=self.wreg, indim=self.dim)

    def defmodel(self):
        xidx, yidx, zidx = T.ivectors("xidx", "yidx", "zidx")
        xemb, yemb = self.embed(xidx, yidx)
        iseq = T.stack(xemb, yemb) # (2, batsize, dims)
        iseq = iseq.dimshuffle(1, 0, 2) # (batsize, 2, dims)
        oseq = self.rnnu(iseq)
        om = oseq[:, np.int32(-1), :] # om is (batsize, dims)
        preds = T.dot(om, self.sm)
        probs = T.nnet.softmax(preds)
        return probs, [xidx, yidx], zidx

    @property
    def depparams(self):
        return self.rnnu.parameters


class AdaXRNNKMSM(XRNNKMSGDSM):
    def __init__(self, rho=0.95, epsilon=0.000001, **kw):
        self.lr = 1.0
        self.rho = rho
        self.epsilon = epsilon
        super(AdaXRNNKMSM, self).__init__(**kw)

    def getupdates(self, params, grads):
        adadelta_egs = [theano.shared(np.zeros_like(param.get_value()).astype("float32")) for param in params]
        adadelta_edxs= [theano.shared(np.zeros_like(param.get_value()).astype("float32")) for param in params]
        updates = []
        for (p, g, eg, ed) in zip(params, grads, adadelta_egs, adadelta_edxs):
            egp = self.rho * eg + (1 - self.rho) * (g**2)
            updates.append((eg, egp.astype("float32")))
            deltap = - (T.sqrt(ed + self.epsilon) / T.sqrt(egp + self.epsilon)) * g
            updates.append((p, (p + self.lr * deltap).astype("float32")))
            edp = self.rho * ed + (1 - self.rho) * (deltap**2)
            updates.append((ed, edp.astype("float32")))
        return updates


class ADDKMSGDMM(SGDBase, Predictor, Profileable, Saveable):
    def __init__(self, dim=10, vocabsize=10, negrate=1, margin=0.9, **kw):
        super(ADDKMSGDMM, self).__init__(**kw)
        self.dim = dim
        self.vocabsize = vocabsize
        self.negrate = negrate
        self.margin = margin
        offset=0.5
        scale=1.
        self.W = theano.shared((np.random.random((self.vocabsize, self.dim)).astype("float32")-offset)*scale, name="W")

    def train(self, trainX, labels, evalinter=10): # X: z, x, y, v OR r, s, o, v
        self.batsize = int(ceil(trainX.shape[0]*1./self.numbats))
        self.tbatsize = theano.shared(np.int32(self.batsize))
        pdot, ndot, inps = self.defmodel()
        tErr = self.geterr(pdot, ndot)
        tReg = self.getreg()
        #embed()
        tCost = tErr + tReg
        showgraph(tCost)
        #embed() # tErr.eval({inps[0]: [0], inps[1]:[10], gold: [1]})

        trainf = self.gettrainf(inps, [tErr, tCost], tCost)
        err = self.trainloop(trainf=self.getbatchloop(trainf, self.getsamplegen(trainX, labels)),
                             evalinter=evalinter,
                             normf=self.getnormf())
        return err

    def defmodel(self):
        xidx, yidx, zidx, nzidx = T.ivectors("xidx", "yidx", "zidx", "nzidx") # rhs corruption only
        xemb, yemb, zemb, nzemb = self.embed(xidx, yidx, zidx, nzidx)
        om = xemb - yemb + zemb
        nom = xemb - yemb + nzemb
        dotp = om.norm(2, axis=1)
        ndotp = nom.norm(2, axis=1)
        #sdotp = T.nnet.sigmoid(dotp)
        #sndotp = T.nnet.sigmoid(ndotp)
        return dotp, ndotp, [xidx, yidx, zidx, nzidx]

    def embed(self, *idxs):
        return tuple(map(lambda x: self.W[x, :], idxs))

    def getreg(self, regf=lambda x: T.sum(x**2), factor=1./2):
        return factor * reduce(lambda x, y: x + y,
                               map(lambda x: regf(x) * self.twreg,
                                   self.ownparams))

    def geterr(self, pdot, ndot): # max margin
        comp = T.clip(self.margin - pdot + ndot, 0, np.infty)
        return T.sum(comp)

    @property
    def ownparams(self):
        return [self.W]

    @property
    def depparams(self):
        return []

    def getsamplegen(self, trainX, labels):
        batsize = self.batsize

        def samplegen():
            nonzeroidx = sorted(np.random.randint(0, trainX.shape[0], size=(batsize,)).astype("int32"))
            trainXsample = trainX[nonzeroidx, :].astype("int32")
            labelsample = labels[nonzeroidx].astype("int32")
            corruptedlabels = np.random.randint(0, self.vocabsize, size=(batsize,)).astype("int32")
            return [trainXsample[:, 0], trainXsample[:, 1], labelsample, corruptedlabels]     # [[s*], [r*], [o*]]
        return samplegen

    def getpredictfunction(self):
        pdot, _, inps = self.defmodel()
        scoref = theano.function(inputs=inps[:3], outputs=pdot)
        def pref(s, r, o):
            args = [np.asarray(i).reshape((1,)).astype("int32") for i in [s, r, o]]
            return scoref(*args)
        return pref

    def getnormf(self):
            return None


class GRUKMSGDMM(ADDKMSGDMM):

    def __init__(self, innerdim=None, **kw):
        super(GRUKMSGDMM, self).__init__(**kw)
        if innerdim is None:
            innerdim = self.dim
        self.innerdim = innerdim
        self.initrnnu()

    def initrnnu(self):
        self.rnnu = GRU(dim=self.dim, wreg=self.wreg, innerdim=self.innerdim)

    def defmodel(self):
        xidx, yidx, zidx, nzidx = T.ivectors("xidx", "yidx", "zidx", "nzidx") # rhs corruption only
        xemb, yemb, zemb, nzemb = self.embed(xidx, yidx, zidx, nzidx)
        iseq = T.stack(xemb, yemb) # (2, batsize, dims)
        iseq = iseq.dimshuffle(1, 0, 2) # (batsize, 2, dims)
        oseq = self.rnnu(iseq)
        om = oseq[:, np.int32(-1), :] # om is (batsize, dims)  ---> last output
        dotp = T.batched_dot(om, zemb)
        ndotp = T.batched_dot(om, nzemb)
        sdotp = T.nnet.sigmoid(dotp)
        sndotp = T.nnet.sigmoid(ndotp)
        return dotp, ndotp, [xidx, yidx, zidx, nzidx]

    @property
    def depparams(self):
        return self.rnnu.parameters

class LSTMKMSGDMM(GRUKMSGDMM):
    def initrnnu(self):
        self.rnnu = LSTM(dim=self.dim, innerdim=self.innerdim, wreg=self.wreg)

def showgraph(var):
    pass
    #theano.printing.pydotprint(var, outfile="/home/denis/logreg_pydotprint_prediction.png", var_with_name_simple=True)

if __name__ == "__main__":
    pass
