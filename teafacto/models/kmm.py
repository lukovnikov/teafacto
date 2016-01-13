from math import ceil

import numpy as np
import theano

from theano import tensor as T

from teafacto.core.rnn import RNUBase
from teafacto.core.trainutil import SGDBase, Saveable, Profileable, Normalizable, Predictor

__author__ = 'denis'

# Knowledge Model - Margin objective
# all models here assume single index space


class KMM(SGDBase, Predictor, Profileable, Saveable):
    def __init__(self, vocabsize=10, numrels=0, negrate=1, margin=1.0, **kw):
        super(KMM, self).__init__(**kw)
        self.vocabsize = vocabsize
        self.negrate = negrate
        self.margin = margin
        self.numrels = numrels

    @property
    def printname(self):
        return super(KMM, self).printname + "+n"+str(self.negrate)

    def defproblem(self):
        pdot, ndot, inps = self.defmodel()
        tErr = self.geterr(pdot, ndot)
        tReg = self.getreg()
        tCost = tErr + tReg
        return inps, tErr, tCost

    def ___train(self, trainX, labels, evalinter=10): # X: z, x, y, v OR r, s, o, v
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
        validf = self.getvalidf(inps, [tErr])
        err = self.trainloop(trainf=self.getbatchloop(trainf, self.getsamplegen(trainX, labels)),
                             evalinter=evalinter,
                             normf=self.getnormf(),
                             validf=validf)
        return err

    def defmodel(self):
        sidx = T.ivector("sidx")
        pathidxs = T.imatrix("pathidxs")
        zidx, nzidx = T.ivectors("zidx", "nzidx") # rhs corruption only
        dotp, ndotp = self.definnermodel(sidx, pathidxs, zidx, nzidx)
        return dotp, ndotp, [sidx, pathidxs, zidx, nzidx]

    def definnermodel(self, sidx, pathidxs, zidx, nzidx):
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

    def getsamplegen(self, data, labels, onebatch=False):
        batsize = self.batsize if not onebatch else data.shape[0]
        negrate = self.negrate

        def samplegen():
            nonzeroidx = sorted(np.random.randint(0, data.shape[0], size=(batsize,)).astype("int32"))
            trainXsample = data[nonzeroidx, :].astype("int32")
            trainXsample = np.repeat(trainXsample, negrate, axis=0)
            labelsample = labels[nonzeroidx].astype("int32")
            labelsample = np.repeat(labelsample, negrate, axis=0)
            corruptedlabels = np.random.randint(0, self.vocabsize, size=(batsize,)).astype("int32")
            for i in range(negrate-1):
                corruptedlabels = np.append(corruptedlabels, np.random.randint(0, self.vocabsize, size=(batsize,)).astype("int32"), axis=0)
            return [trainXsample[:, 0], trainXsample[:, 1:], labelsample, corruptedlabels]     # start, path, target, bad_target
        return samplegen

    def getpredictfunction(self):
        pdot, _, inps = self.defmodel()
        scoref = theano.function(inputs=[inps[0], inps[1], inps[2]], outputs=pdot)
        def pref(s, path, o):
            args = [np.asarray(i).astype("int32") for i in [s, path, o]]
            return scoref(*args)
        return pref


class EKMM(KMM, Normalizable):
    def __init__(self, dim=10, **kw):
        super(EKMM, self).__init__(**kw)
        offset = 0.5
        scale = 1.
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
        if len(idxs) == 1:
            return self.W[idxs[0], :]
        else:
            return tuple(map(lambda x: self.W[x, :], idxs))

    def definnermodel(self, sidx, pathidxs, zidx, nzidx):#pathemb: (batsize, seqlen), *emb: (batsize)
        om, _ = theano.scan(fn=self.traverse,
                         sequences=pathidxs.T, # --> (seqlen, batsize)
                         outputs_info=[None, self.embed(sidx)]
                         )
        om = om[0] # --> (seqlen, batsize, dim)
        om = om[-1, :, :] # --> (batsize, dim)
        dotp = self.membership(om, self.embed(zidx))
        ndotp = self.membership(om, self.embed(nzidx))
        return dotp, ndotp

    def traverse(self, x_t, h_tm1):
        raise NotImplementedError("use subclass")

    def membership(self, h_tm1, t):
        raise NotImplementedError("use subclass")


class DistMemberEKMM(EKMM):
    def membership(self, o, t):
        return -T.sum(T.sqr(o - t), axis=1)


class DotMemberEKMM(EKMM):
    def membership(self, o, t):
        return T.batched_dot(o, t)


class CosMemberEKMM(EKMM):
    def membership(self, o, t):
        return T.batched_dot(o, t) / (o.norm(2, axis=1) * t.norm(2, axis=1))


class AddEKMM(DistMemberEKMM):                # TransE
    def traverse(self, x_t, h_tm1): # x_t: (batsize, dim), h_tm1: (batsize, dim)
        h = h_tm1 + self.embed(x_t)
        return [h, h]


class VecMulEKMM(DotMemberEKMM):    # Bilinear Diag
    def traverse(self, x_t, h_tm1): # x_t: (batsize, dim), h_tm1: (batsize, dim)
        h = self.embed(x_t) * h_tm1
        return [h, h]


class VecMulEKMMDist(DistMemberEKMM, VecMulEKMM):
    pass


class MatMulEKMM(DotMemberEKMM):    # RESCAL
    def __init__(self, **kw):
        super(MatMulEKMM, self).__init__(**kw)
        offset = 0.5
        scale = 1.
        self.R = theano.shared((np.random.random((self.numrels, self.dim, self.dim)).astype("float32")-offset)*scale, name="R")

    @property
    def ownparams(self):
        return super(MatMulEKMM, self).ownparams + [self.R]

    def traverse(self, x_t, h_tm1): # x_t : (batsize, dim, dim), h_tm1 : (batsize, dim)
        h = T.batched_dot(self.embedR(x_t-self.vocabsize+self.numrels), h_tm1)
        return [h, h]

    def embedR(self, idxs): # pathidxs: (batsize)
        return self.R[idxs, :] # return: (batsize, dim, dim)


class MatMulEKMMCos(CosMemberEKMM, MatMulEKMM):
    pass

class TransAddEKMM(DotMemberEKMM):
    def __init__(self, innerdim=10, **kw):
        super(TransAddEKMM, self).__init__(**kw)
        offset = 0.5
        scale = 1.
        self.innerdim = innerdim
        self.Rtrans = theano.shared((np.random.random((self.numrels, self.dim, self.innerdim)).astype("float32")-offset)*scale, name="Rtrans")
        self.Radd = theano.shared((np.random.random((self.numrels, self.innerdim)).astype("float32")-offset)*scale, name="Radd")
        self.Rtransinv = theano.shared((np.random.random((self.numrels, self.innerdim, self.dim)).astype("float32")-offset)*scale, name="Rtransinv")

    @property
    def ownparams(self):
        return super(TransAddEKMM, self).ownparams + [self.Rtrans, self.Radd, self.Rtransinv]

    def traverse(self, x_t, h_tm1):
        x_t = x_t - self.vocabsize + self.numrels
        h = T.batched_dot(T.batched_dot(h_tm1, self.Rtrans[x_t, :]) + self.Radd[x_t, :], self.Rtransinv[x_t, :])
        return [h, h]

class RNNEKMM(DotMemberEKMM):

    def traverse(self, x_t, h_tm1):         # x_t: (batsize, dim), h_tm1: (batsize, dim)
        return self.rnnu.rec(self.embed(x_t), h_tm1)

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

class ERNNEKMM(RNNEKMM):
    def traverse(self, x_t, h_tm1):
        return self.rnnu.rec(x_t - self.vocabsize + self.numrels, h_tm1)


class RNNEOKMM(RNNEKMM):    # is this still useful? TODO
    def onrnnudefined(self):
        self.initwout()

    def initwout(self):
        offset = 0.5
        scale = 0.1
        self.Wout = theano.shared((np.random.random((self.rnnu.innerdim, self.dim)).astype("float32")-offset)*scale, name="Wout")

    def membership(self, o, t):
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