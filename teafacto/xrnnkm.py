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
import sys, os, cPickle as pickle, inspect
from rnnkm import Saveable, Profileable, Predictor, Normalizable
from rnn import GRU, LSTM, RNUBase
from optimizers import SGD, RMSProp, AdaDelta, Optimizer

class SGDBase(object):
    def __init__(self, maxiter=50, lr=0.0001, numbats=100, wreg=0.00001, **kw):
        self.maxiter = maxiter
        self.numbats = numbats
        self.wreg = wreg
        self.tnumbats = theano.shared(np.float32(self.numbats), name="numbats")
        self.twreg = theano.shared(np.float32(self.wreg), name="wreg")
        self._optimizer = SGD(lr)
        super(SGDBase, self).__init__(**kw)

    @property
    def printname(self):
        return self.__class__.__name__ + "+" + self._optimizer.__class__.__name__

    def __add__(self, other):
        if isinstance(other, Optimizer):
            self._optimizer = other
            other.onattach(self)
            return self
        else:
            raise Exception("unknown type of composition argument")

    def gettrainf(self, finps, fouts, cost):
        params = self.ownparams + self.depparams
        grads = T.grad(cost, wrt=params)
        updates = self.getupdates(params, grads)
        #showgraph(updates[0][1])
        return theano.function(inputs=finps,
                               outputs=fouts,
                               updates=updates,
                               profile=self._profiletheano)

    def getupdates(self, params, grads):
        return self._optimizer.getupdates(params, grads)

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


class KMSM(SGDBase, Saveable, Profileable, Predictor, Normalizable):
    def __init__(self, vocabsize=10, negrate=None, margin=None, **kw):
        super(KMSM, self).__init__(**kw)
        self.vocabsize = vocabsize

    def train(self, trainX, labels, evalinter=10):
        self.batsize = int(ceil(trainX.shape[0]*1./self.numbats))
        self.tbatsize = theano.shared(np.int32(self.batsize))
        probs, gold, inps = self.defmodel()
        tErr = self.geterr(probs, gold)
        tReg = self.getreg()
        #embed()
        tCost = tErr + tReg
        #showgraph(tCost)
        #embed() # tErr.eval({inps[0]: [0], inps[1]:[10], gold: [1]})

        trainf = self.gettrainf(inps, [tErr, tCost], tCost)
        err = self.trainloop(trainf=self.getbatchloop(trainf, self.getsamplegen(trainX, labels)),
                             evalinter=evalinter,
                             normf=self.getnormf())
        return err

    def defmodel(self):
        pathidxs = T.imatrix("pathidxs")
        zidx = T.ivector("zidx") # rhs corruption only
        scores = self.definnermodel(pathidxs) # ? scores: float(batsize, vocabsize)
        probs = T.nnet.softmax(scores) # row-wise softmax, ? probs: float(batsize, vocabsize)
        return probs, zidx, [pathidxs, zidx]

    def definnermodel(self, pathidxs):
        raise NotImplementedError("use subclass")

    def getreg(self, regf=lambda x: T.sum(x**2), factor=1./2):
        return factor * reduce(lambda x, y: x + y,
                               map(lambda x: regf(x) * self.twreg,
                                   self.ownparams))

    def geterr(self, probs, gold): # cross-entropy
        return -T.mean(T.log(probs[T.arange(self.batsize), gold]))

    @property
    def ownparams(self):
        return []

    @property
    def depparams(self):
        return []

    def getnormf(self):
        return None

    def getsamplegen(self, trainX, labels):
        batsize = self.batsize

        def samplegen():
            nonzeroidx = sorted(np.random.randint(0, trainX.shape[0], size=(batsize,)).astype("int32"))
            trainXsample = trainX[nonzeroidx, :].astype("int32")
            labelsample = labels[nonzeroidx].astype("int32")
            return [trainXsample, labelsample]     # start + path, target, bad_target
        return samplegen

    def getpredictfunction(self):
        probs, gold, inps = self.defmodel()
        score = probs[T.arange(gold.shape[0]), gold]
        scoref = theano.function(inputs=[inps[0], inps[1]], outputs=score)
        def pref(path, o):
            args = [np.asarray(i).astype("int32") for i in [path, o]]
            return scoref(*args)
        return pref


class SMSM(KMSM):

    def defmodel(self):
        pathidxs = T.imatrix("pathidxs")  # integers of (batsize, seqlen)
        zidxs = T.imatrix("zidxs") # integers of (batsize, seqlen)
        scores = self.definnermodel(pathidxs) #predictions, floats of (batsize, seqlen, vocabsize)
        #probs = T.nnet.softmax(scores) # row-wise softmax; probs: (batsize, seqlen, vocabsize) #softmax doesn't work on tensor3D
        probs = theano.scan(fn=T.nnet.softmax,
                            sequences=scores,
                            outputs_info=[None])
        return probs, zidxs, [pathidxs, zidxs]

    def geterr(self, probs, golds): # cross-entropy; probs: floats of (batsize, seqlen, vocabsize), gold: indexes of (batsize, seqlen)
        return -T.mean(
                    T.log(
                        probs[T.arange(probs.shape[0])[:, None],
                              T.arange(probs.shape[1])[None, :],
                              golds])) # --> prob: floats of (batsize, seqlen) #TODO: is mean of logs of all matrix elements correct?

    def getsamplegen(self, trainX, labels): # trainX and labels must be of same dimensions
        batsize = self.batsize

        def samplegen():
            nonzeroidx = sorted(np.random.randint(0, trainX.shape[0], size=(batsize,)).astype("int32"))
            trainXsample = trainX[nonzeroidx, :].astype("int32")
            labelsample = labels[nonzeroidx, :].astype("int32")
            return [trainXsample, labelsample]     # input seq, output seq
        return samplegen

    def getpredictfunction(self):
        probs, golds, inps = self.defmodel()
        score = probs[T.arange(golds.shape[0]), golds]
        scoref = theano.function(inputs=[inps[0], inps[1]], outputs=score)
        def pref(path, o):
            args = [np.asarray(i).astype("int32") for i in [path, o]]
            return scoref(*args)
        return pref

class ESMSM(SMSM, Normalizable): # identical to EKMSM since the same prediction part
    def __init__(self, dim=10, **kw):
        super(ESMSM, self).__init__(**kw)
        offset=0.5
        scale=1.
        self.dim = dim
        self.W = theano.shared((np.random.random((self.vocabsize, self.dim)).astype("float32")-offset)*scale, name="W")

    def getnormf(self):
        if self._normalize is True:
            norms = self.W.norm(2, axis=1).reshape((self.W.shape[0], 1))
            upd = (self.W, self.W/norms)
            return theano.function(inputs=[], outputs=[], updates=[upd])
        else:
            return None

    @property
    def printname(self):
        return super(ESMSM, self).printname + "+E" + str(self.dim)+"D"

    @property
    def ownparams(self):
        return [self.W]

    @property
    def depparams(self):
        return []

    def embed(self, *idxs):
        return tuple(map(lambda x: self.W[x, :], idxs))

    def definnermodel(self, pathidxs):
        pathembs = self.embed(pathidxs) # pathembs: (batsize, seqlen, edim); zemb: (batsize, edim)
        return self.innermodel(pathembs)

    def innermodel(self, pathembs):
        raise NotImplementedError("use subclass")

class RNNESMSM(ESMSM): # identical to RNNEKMSM since same prediction part

    def innermodel(self, pathembs): #pathemb: (batsize, seqlen, dim)
        oseq = self.rnnu(pathembs)
        om = oseq[:, -1, :] # om is (batsize, innerdims)  ---> last output
        scores = T.dot(om, self.Wout) # --> (batsize, vocabsize)
        return scores

    @property
    def printname(self):
        return super(RNNESMSM, self).printname + "+" + self.rnnu.__class__.__name__+ ":" + str(self.rnnu.innerdim) + "D"

    @property
    def depparams(self):
        return self.rnnu.parameters

    def __add__(self, other):
        if isinstance(other, RNUBase):
            self.rnnu = other
            self.onrnnudefined()
            return self
        else:
            return super(RNNESMSM, self).__add__(other)


    @property
    def ownparams(self):
        return super(RNNESMSM, self).ownparams + [self.Wout]

    def onrnnudefined(self):
        self.initwout()

    def initwout(self):
        offset = 0.5
        scale = 0.1
        self.Wout = theano.shared((np.random.random((self.rnnu.innerdim, self.vocabsize)).astype("float32")-offset)*scale, name="Wout")


class EKMSM(KMSM, Normalizable):
    def __init__(self, dim=10, **kw):
        super(EKMSM, self).__init__(**kw)
        offset=0.5
        scale=1.
        self.dim = dim
        self.W = theano.shared((np.random.random((self.vocabsize, self.dim)).astype("float32")-offset)*scale, name="W")

    def getnormf(self):
        if self._normalize is True:
            norms = self.W.norm(2, axis=1).reshape((self.W.shape[0], 1))
            upd = (self.W, self.W/norms)
            return theano.function(inputs=[], outputs=[], updates=[upd])
        else:
            return None

    @property
    def printname(self):
        return super(EKMSM, self).printname + "+E" + str(self.dim)+"D"

    @property
    def ownparams(self):
        return [self.W]

    @property
    def depparams(self):
        return []

    def embed(self, *idxs):
        return tuple(map(lambda x: self.W[x, :], idxs))

    def definnermodel(self, pathidxs):
        pathembs, = self.embed(pathidxs) # pathembs: (batsize, seqlen, edim); zemb: (batsize, edim)
        return self.innermodel(pathembs)

    def innermodel(self, pathembs):
        raise NotImplementedError("use subclass")

class RNNEKMSM(EKMSM):

    def innermodel(self, pathembs): #pathemb: (batsize, seqlen, dim)
        oseq = self.rnnu(pathembs)
        om = oseq[:, -1, :] # om is (batsize, innerdims)  ---> last output
        scores = T.dot(om, self.Wout) # --> (batsize, vocabsize)
        return scores

    @property
    def printname(self):
        return super(RNNEKMSM, self).printname + "+" + self.rnnu.__class__.__name__+ ":" + str(self.rnnu.innerdim) + "D"

    @property
    def depparams(self):
        return self.rnnu.parameters

    def __add__(self, other):
        if isinstance(other, RNUBase):
            self.rnnu = other
            self.onrnnudefined()
            return self
        else:
            return super(EKMSM, self).__add__(other)


    @property
    def ownparams(self):
        return super(RNNEKMSM, self).ownparams + [self.Wout]

    def onrnnudefined(self):
        self.initwout()

    def initwout(self):
        offset = 0.5
        scale = 0.1
        self.Wout = theano.shared((np.random.random((self.rnnu.innerdim, self.vocabsize)).astype("float32")-offset)*scale, name="Wout")



class KMM(SGDBase, Predictor, Profileable, Saveable):
    def __init__(self, vocabsize=10, negrate=1, margin=1.0, **kw):
        super(KMM, self).__init__(**kw)
        self.vocabsize = vocabsize
        self.negrate = negrate
        self.margin = margin

    @property
    def printname(self):
        return super(KMM, self).printname + "+n"+str(self.negrate)

    def train(self, trainX, labels, evalinter=10): # X: z, x, y, v OR r, s, o, v
        self.batsize = int(ceil(trainX.shape[0]*1./self.numbats))
        self.tbatsize = theano.shared(np.int32(self.batsize))
        pdot, ndot, inps = self.defmodel()
        tErr = self.geterr(pdot, ndot)
        tReg = self.getreg()
        #embed()
        tCost = tErr + tReg
        #showgraph(tCost)
        #embed() # tErr.eval({inps[0]: [0], inps[1]:[10], gold: [1]})

        trainf = self.gettrainf(inps, [tErr, tCost], tCost)
        err = self.trainloop(trainf=self.getbatchloop(trainf, self.getsamplegen(trainX, labels)),
                             evalinter=evalinter,
                             normf=self.getnormf())
        return err

    def defmodel(self):
        pathidxs = T.imatrix("pathidxs")
        zidx, nzidx = T.ivectors("zidx", "nzidx") # rhs corruption only
        dotp, ndotp = self.definnermodel(pathidxs, zidx, nzidx)
        return dotp, ndotp, [pathidxs, zidx, nzidx]

    def definnermodel(self, pathidxs, zidx, nzidx):
        raise NotImplementedError("use subclass")

    def getreg(self, regf=lambda x: T.sum(x**2), factor=1./2):
        return factor * reduce(lambda x, y: x + y,
                               map(lambda x: regf(x) * self.twreg,
                                   self.ownparams))

    def geterr(self, pdot, ndot): # max margin
        comp = T.clip(self.margin - pdot + ndot, 0, np.infty)
        return T.sum(comp)

    @property
    def ownparams(self):
        return []

    @property
    def depparams(self):
        return []

    def getnormf(self):
        return None

    def getsamplegen(self, trainX, labels):
        batsize = self.batsize
        negrate = self.negrate

        def samplegen():
            nonzeroidx = sorted(np.random.randint(0, trainX.shape[0], size=(batsize,)).astype("int32"))
            trainXsample = trainX[nonzeroidx, :].astype("int32")
            trainXsample = np.repeat(trainXsample, negrate, axis=0)
            labelsample = labels[nonzeroidx].astype("int32")
            labelsample = np.repeat(labelsample, negrate, axis=0)
            corruptedlabels = np.random.randint(0, self.vocabsize, size=(batsize,)).astype("int32")
            for i in range(negrate-1):
                corruptedlabels = np.append(corruptedlabels, np.random.randint(0, self.vocabsize, size=(batsize,)).astype("int32"), axis=0)
            return [trainXsample, labelsample, corruptedlabels]     # start + path, target, bad_target
        return samplegen

    def getpredictfunction(self):
        pdot, _, inps = self.defmodel()
        scoref = theano.function(inputs=[inps[0], inps[1]], outputs=pdot)
        def pref(path, o):
            args = [np.asarray(i).astype("int32") for i in [path, o]]
            return scoref(*args)
        return pref


class EKMM(KMM, Normalizable):
    def __init__(self, dim=10, **kw):
        super(EKMM, self).__init__(**kw)
        offset=0.5
        scale=1.
        self.dim = dim
        self.W = theano.shared((np.random.random((self.vocabsize, self.dim)).astype("float32")-offset)*scale, name="W")

    def getnormf(self):
        if self._normalize is True:
            norms = self.W.norm(2, axis=1).reshape((self.W.shape[0], 1))
            upd = (self.W, self.W/norms)
            return theano.function(inputs=[], outputs=[], updates=[upd])
        else:
            return None

    @property
    def printname(self):
        return super(EKMM, self).printname + "+E" + str(self.dim)+"D"

    @property
    def ownparams(self):
        return [self.W]

    @property
    def depparams(self):
        return []

    def embed(self, *idxs):
        return tuple(map(lambda x: self.W[x, :], idxs))

    def definnermodel(self, pathidxs, zidx, nzidx):
        pathembs, zemb, nzemb = self.embed(pathidxs, zidx, nzidx)
        return self.innermodel(pathembs, zemb, nzemb)

    def innermodel(self, pathembs, zemb, nzemb):
        raise NotImplementedError("use subclass")


class AddEKMM(EKMM):

    def innermodel(self, pathembs, zemb, nzemb): #pathemb: (batsize, seqlen, dim)
        om, _ = theano.scan(fn=self.traverse,
                         sequences=pathembs.dimshuffle(1, 0, 2), # --> (seqlen, batsize, dim)
                         outputs_info=[None, self.emptystate(pathembs)] # zeroes like (batsize, dim)
                         )
        om = om[0] # --> (seqlen, batsize, dim)
        om = om[-1, :, :] # --> (batsize, dim)
        dotp = self.membership(om, zemb)
        ndotp = self.membership(om, nzemb)
        return dotp, ndotp

    def emptystate(self, pathembs):
        return T.zeros_like(pathembs[:, 0, :])

    def traverse(self, x_t, h_tm1):
        h = h_tm1 + x_t
        return [h, h]

    def membership(self, o, t):
        return -T.sum(T.sqr(o - t), axis=1)


class MulEKMM(AddEKMM):
    def traverse(self, x_t, h_tm1):
        h = x_t * h_tm1
        return [h, h]

    def membership(self, o, t):
        return T.batched_dot(o, t)

    def emptystate(self, pathembs):
        return T.ones_like(pathembs[:, 0, :])


class RNNEKMM(EKMM):

    def innermodel(self, pathembs, zemb, nzemb): #pathemb: (batsize, seqlen, dim)
        oseq = self.rnnu(pathembs)
        om = oseq[:, -1, :] # om is (batsize, innerdims)  ---> last output
        dotp = self.membership_dot(om, zemb)
        ndotp = self.membership_dot(om, nzemb)
        return dotp, ndotp

    def membership_dot(self, o, t):
        return T.batched_dot(o, t)

    def membership_add(self, o, t):
        return -T.sum(T.sqr(o - t), axis=1)

    @property
    def printname(self):
        return super(RNNEKMM, self).printname + "+" + self.rnnu.__class__.__name__

    @property
    def depparams(self):
        return self.rnnu.parameters

    def __add__(self, other):
        if isinstance(other, RNUBase):
            self.rnnu = other
            self.onrnnudefined()
            return self
        else:
            return super(EKMM, self).__add__(other)

    def onrnnudefined(self):
        pass


class RNNEOKMM(RNNEKMM):
    def onrnnudefined(self):
        self.initwout()

    def initwout(self):
        offset = 0.5
        scale = 0.1
        self.Wout = theano.shared((np.random.random((self.rnnu.innerdim, self.dim)).astype("float32")-offset)*scale, name="Wout")

    def membership_dot(self, o, t):
        om = T.dot(o, self.Wout)
        return T.batched_dot(om, t)

    def membership_add(self, o, t):
        om = T.dot(o, self.Wout)
        return -T.sum(T.sqr(om - t), axis=1)

    @property
    def ownparams(self):
        return super(RNNEOKMM, self).ownparams + [self.Wout]

    @property
    def printname(self):
        return super(RNNEKMM, self).printname + "+" + self.rnnu.__class__.__name__ + ":" + str(self.rnnu.innerdim) + "D"


def showgraph(var):
    pass
    #theano.printing.pydotprint(var, outfile="/home/denis/logreg_pydotprint_prediction.png", var_with_name_simple=True)

if __name__ == "__main__":
    pass
