from math import ceil
import numpy as np
import theano
from theano import tensor as T
from teafacto.rnn import RNUBase
from teafacto.xxrnnkm import Predictor, Profileable, Saveable, Normalizable
from teafacto.km import SGDBase, Saveable, Profileable, Normalizable, Predictor

__author__ = 'denis'

# Knowledge Model - Margin objective


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
            return [trainXsample[:, 0], trainXsample[:, 1:], labelsample, corruptedlabels]     # start, path, target, bad_target
        return samplegen

    def getpredictfunction(self):
        pdot, _, inps = self.defmodel()
        scoref = theano.function(inputs=[inps[0], inps[1], inps[2]], outputs=pdot)
        def pref(path, o):
            args = [np.asarray(i).astype("int32") for i in [path, o]]
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
        return tuple(map(lambda x: self.W[x, :], idxs))

    def embedpath(self, path):
        return self.embed(path)

    def definnermodel(self, sidx, pathidxs, zidx, nzidx):
        semb, zemb, nzemb = self.embed(sidx, zidx, nzidx)
        pathembs = self.embedpath(pathidxs)
        return self.innermodel(semb, pathembs, zemb, nzemb)

    def innermodel(self, semb, pathembs, zemb, nzemb):
        raise NotImplementedError("use subclass")


class AddEKMM(EKMM):
    # TransE

    def innermodel(self, semb, pathembs, zemb, nzemb): #pathemb: (batsize, seqlen, dim)
        om, _ = theano.scan(fn=self.traverse,
                         sequences=self._scanshuffle(pathembs), # --> (seqlen, batsize, dim{1,2})
                         outputs_info=[None, semb] # zeroes like (batsize, dim)
                         )
        om = om[0] # --> (seqlen, batsize, dim)
        om = om[-1, :, :] # --> (batsize, dim)
        dotp = self.membership(om, zemb)
        ndotp = self.membership(om, nzemb)
        return dotp, ndotp

    def _scanshuffle(self, pathembs):
        return pathembs.dimshuffle(1, 0, 2)

    def traverse(self, x_t, h_tm1):
        h = h_tm1 + x_t
        return [h, h]

    def membership(self, o, t):
        return -T.sum(T.sqr(o - t), axis=1)


class DiagMulEKMM(AddEKMM):
    # Bilinear Diag
    def traverse(self, x_t, h_tm1):
        h = x_t * h_tm1
        return [h, h]

    def membership(self, o, t):
        return T.batched_dot(o, t)


class MulEKMM(DiagMulEKMM):
    # RESCAL
    # TODO: TEST (test all EKMM's)

    def __init__(self, **kw):
        super(MulEKMM, self).__init__(**kw)
        offset = 0.5
        scale = 1.
        self.R = theano.shared((np.random.random((self.vocabsize, self.dim, self.dim)).astype("float32")-offset)*scale, name="R")

    @property
    def ownparams(self):
        return super(MulEKMM, self).ownparams + [self.R]

    def traverse(self, x_t, h_tm1): # x_t : (batsize, dim, dim), h_tm1 : (batsize, dim)
        h = T.batched_dot(x_t, h_tm1)
        return [h, h]

    def embedpath(self, pathidxs): # pathidxs: (batsize, seqlen)
        return self.R[pathidxs, :] # return: (batsize, seqlen, dim, dim)

    def _scanshuffle(self, pathembs):
        return pathembs.dimshuffle(1, 0, 2, 3) # ==> (seqlen, batsize, dim, dim)


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