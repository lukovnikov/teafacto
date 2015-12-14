__author__ = 'denis'
import theano
from theano import tensor as T
import numpy as np
from tf import TFSGD
from utils import *
from math import ceil, floor
from datetime import datetime as dt
from IPython import embed
import sys, os, cPickle as pickle

class GRU(object):
    def __init__(self,
                 dim=20,
                 indim=20,
                 wreg=0.0001,
                 initmult=0.01,
                 gateactivation=T.nnet.sigmoid,
                 outpactivation=T.tanh,
                 **kw):
        super(GRU, self).__init__(**kw)
        self.dim = dim
        self.indim = indim
        self.wreg = wreg
        self.initmult = initmult
        self.gateactivation = gateactivation
        self.outpactivation = outpactivation

        self.initparams()

    def initparams(self):
        self.paramnames = ["uz", "wz", "ur", "wr", "u", "w", "br", "bz", "b"]
        params = {}
        indim = self.indim
        for paramname in self.paramnames:
            if paramname[0] == "b":
                shape = (self.dim,)
            elif paramname[0] == "w":
                shape = (indim, self.dim)
            else:
                shape = (self.dim, self.dim)
            paramval = np.random.random(shape).astype("float32")*self.initmult
            params[paramname] = theano.shared(paramval)
            setattr(self, paramname, params[paramname])
        self.initstate = T.zeros((self.indim,), dtype="float32")

    def _getz(self, x_t, h_tm1):
        return self.gateactivation(T.dot(h_tm1, self.uz) + T.dot(x_t, self.wz) + self.bz)
    def _getr(self, x_t, h_tm1):
        return self.gateactivation(T.dot(h_tm1, self.ur) + T.dot(x_t, self.wr) + self.br)
    def _gethh(self, x_t, h_tm1, r):
        return self.outpactivation(T.dot(h_tm1 * r, self.u) + T.dot(x_t, self.w) + self.b)
    def _geth(self, z, hh, h_tm1):
        return z * h_tm1 + (1-z) * hh

    def rec(self, x_t, h_tm1):
        '''
        :param x_t: input values (nb_samples, nb_feats) for this recurrence step
        :param h_tm1: previous states (nb_samples, out_dim)
        :return: new state (nb_samples, out_dim)
        '''
        z = self._getz(x_t, h_tm1)
        r = self._getr(x_t, h_tm1)
        hh = self._gethh(x_t, h_tm1, r)
        h = self._geth(z, hh, h_tm1)
        return h

    def recur(self, x):
        inputs = x.dimshuffle(1, 0, 2) # inputs is (seq_len, nb_samples, nb_feats)
        outputs, _ = theano.scan(fn=self.rec,
                                 sequences=inputs,
                                 outputs_info=
                                    T.tile(self.initstate.reshape((1, self.initstate.shape[0])), [inputs.shape[1], 1]),
                                 n_steps=inputs.shape[0])
        return outputs.dimshuffle(1, 0, 2) #output is (nb_samples, seq_len, nb_feats)

    def __call__(self, x):
        return self.getoutput(x)

    def getoutput(self, x):
        '''
        :param x: symbolic input tensor for shape (nb_samples, seq_len, nb_feats) where
            nb_samples is the number of samples (number of sequences) in the current input
            seq_len is the maximum length of the sequences
            nb_feats is the number of features per sequence element
        :return: symbolic output tensor for shape (nb_samples, seq_len, out_dim) where
            nb_samples is the number of samples (number of sequences) in the original input
            seq_len is the maximum length of the sequences
            out_dim is the dimension of the output vector as specified by the dim argument in the constructor
        '''
        return self.recur(x)

    def getreg(self):
        def regf(x):
            return T.sum(x**2)
        reg = (1./2.) * reduce(lambda x, y: x+y, map(lambda x: regf(getattr(self, x))*self.wreg, self.paramnames))
        return reg

    def getparams(self):
        return map(lambda x: getattr(self, x), self.paramnames)


class Saveable(object):
    def __init__(self, **kw):
        super(Saveable, self).__init__(**kw)
        self._autosave = False
        self._autosave_filepath = None
    ############# Saving and Loading #################"
    def getsavepath(self):
        dfile = os.path.join(os.path.dirname(__file__), "../models/%s.%s" %
                             (os.path.splitext(self.__class__.__name__)[0], dt.now().strftime("%Y-%m-%d=%H:%M")))
        return dfile

    def save(self, filepath=None, extra=None):
        if self._autosave_filepath is not None:
            filepath = self._autosave_filepath
        if filepath is None:
            self._autosave_filepath = self.getsavepath()+".auto"
            filepath = self._autosave_filepath
        with open(filepath, "w") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath) as f:
            ret = pickle.load(f)
        return ret

    @property
    def autosave(self): # for saving after each iter
        self._autosave = True
        return self

class Profileable(object):
    def __init__(self, **kw):
        super(Profileable, self).__init__(**kw)
        self._profiletheano = False
    ############## PROFILING #######################
    @property
    def profiletheano(self):
        self._profiletheano = True
        return self

class Parameterized(object):
    @property
    def parameters(self):
        return self.depparameters + self.ownparameters

    @property
    def ownparameters(self):
        raise NotImplementedError("use sublcass")

    @property
    def depparameters(self):
        raise NotImplementedError("use subclass")

class Normalizable(Parameterized):
    def __init__(self, **kw):
        super(Normalizable, self).__init__(**kw)
        self._normalize = False

    @property
    def normalize(self):
        self._normalize = True
        return self

    def getnormf(self):
        raise NotImplementedError("use concrete subclass")

class Trainable(object):
    def __init__(self, **kw):
        super(Trainable, self).__init__(**kw)

    def train(self, X, evalinter=10):
        raise NotImplementedError("call a subclass")

class Predictor(object):
    def __init__(self, **kw):
        super(Predictor, self).__init__(**kw)
        self.predictfunction = None

    def predict(self, *args):
        if self.predictfunction is None:
            self.predictfunction = self.getpredictfunction()
        return self.predictfunction(*args)

class GradientDescent(Parameterized, Profileable):
    def __init__(self, maxiter=50, lr=0.0001, **kw):
        super(GradientDescent, self).__init__(**kw)
        self.maxiter = maxiter
        self.lr = lr

    def trainloop(self, X, trainf, validf=None, evalinter=1, normf=None):
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
            if normf:
                normf()
            if itercount == self.maxiter:
                stop = True
            itercount += 1
            if erre is None \
                    and validf is not None \
                    and (evalinter != 0 and evalinter != np.infty) \
                    and evalcount == evalinter:
                error = validf(X)
                err.append(error)
                print(error)
                evalcount = 0
            else:
                err.append(erre)
                print(erre)
            print("iter done in %f seconds" % (dt.now() - start).total_seconds())
            evalcount += 1
            if self._autosave:
                self.save()
        return err

    def gettrainf(self, inps, outps, tCost):
        # get gradients
        params = self.parameters
        grads = T.grad(tCost, params)
        updates = self.getupdates(params, grads)
        trainf = theano.function(
            inputs=inps,
            outputs=outps,
            updates=updates,
            profile=self._profiletheano
        )
        return trainf

    def getupdates(self, params, grads):
        return map(lambda (p, g): (p, (p - self.lr * self.numbats * g).astype("float32")), zip(params, grads))

class Batched(object):
    def __init__(self, numbats=100, **kw):
        super(Batched, self).__init__(**kw)
        self.numbats=numbats

    def getbatsize(self, X):
        numsam = X.count_nonzero()
        batsize = int(ceil(numsam*1./self.numbats))
        return batsize

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

class AdaDelta(GradientDescent):
    def __init__(self, rho=0.95, epsilon=0.000001, **kw):
        super(AdaDelta, self).__init__(**kw)
        self.rho = rho
        self.epsilon=epsilon

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


class Regularizable(Parameterized):
    def __init__(self, wreg=0.001, **kw):
        super(Regularizable, self).__init__(**kw)
        self.wreg = wreg

    def getreg(self):
        return self.getownreg() + self.getdepreg()

    def getownreg(self):
        raise NotImplementedError("use concrete subclass")

    def getdepreg(self):
        raise NotImplementedError("use concrete subclass")

class Embedder(Regularizable, Normalizable):
    def __init__(self, vocabsize=10, dims=10, **kw):
        super(Embedder, self).__init__(**kw)
        self.vocabsize = vocabsize
        self.dims = dims
        offset = 0.5
        scaler = 0.1
        self.W = theano.shared(np.random.random((self.vocabsize, self.dims)).astype("float32")*scaler - offset)

    @property
    def depparameters(self):
        return []

    @property
    def ownparameters(self):
        return [self.W]

    def getownreg(self):
        return (1./2.) * (T.sum(self.W**2) * self.wreg)

    def getdepreg(self):
        return self.rnnu.getreg()

    def getnormf(self):
        if self._normalize is True:
            norms = self.W.norm(2, axis=1).reshape((self.W.shape[0], 1))
            upd = (self.W, self.W/norms)
            return theano.function(inputs=[], outputs=[], updates=[upd])
        else:
            return None

    def embed(self, idx):
        return self.W[idx, :]


class RNNTFSGDC(Embedder, Trainable, Predictor, GradientDescent, Batched, Saveable):
    def __init__(self, rnnuc=GRU, dims=10, vocabsize=10, wreg=0.0, negrate=1, corruption="rhs", **kw):
        pass
        super(RNNTFSGDC, self).__init__(**kw)
        self.dims = dims
        self.vocabsize = vocabsize
        self.rnnuc = rnnuc
        self.wreg = wreg
        self.negrate = negrate
        self.corruption = corruption

        offset = 0.5
        self.W = theano.shared(np.random.random((self.vocabsize, self.dims)).astype("float32") - offset)
        self.rnnu = self.rnnuc(dim=self.dims, indim=self.dims, wreg=self.wreg)

    def train(self, X, evalinter=10): # X: z, x, y, v OR r, s, o
        batsize = self.getbatsize(X)
        outps, inps = self.defmodel()
        tErr = self.geterr(*outps)
        tReg = self.getreg(*inps)
        tCost = tErr + tReg
        trainf = self.gettrainf(inps, [tErr, tCost], tCost)
        err = self.trainloop(X=X,
                             trainf=self.getbatchloop(trainf, self.getsamplegen(X, batsize)),
                             evalinter=evalinter,
                             normf=self.getnormf())
        return err

    def geterr(self, x, y):
        return T.sum(y - x)

    def getreg(self, *inp): # l2 regularization
        return (1./2.) * (T.sum(self.W**2) * self.wreg) + self.rnnu.getreg()

    def defmodel(self):
        winp, rinp, hinp = T.ivectors("winp", "rinp", "hinp")
        nwinp, nrinp, nhinp = T.ivectors("nwinp", "nrinp", "nhinp")
        dotp = self.builddot(winp, rinp, hinp, self.rnnu)
        ndotp = self.builddot(nwinp, nrinp, nhinp, self.rnnu)
        dotp = dotp.reshape((dotp.shape[0], 1))
        ndotp = ndotp.reshape((ndotp.shape[0], 1))
        return [dotp, ndotp], [rinp, winp, hinp, nrinp, nwinp, nhinp]


    def builddot(self, winp, rinp, hinp, rnnu):
        hemb = self.W[hinp, :]
        om = self.prebuilddot(winp, rinp, rnnu)
        omdot = T.sum(om * hemb, axis=1)
        return T.nnet.sigmoid(omdot)

    def prebuilddot(self, winp, rinp, rnnu):
        wemb = self.W[winp, :] # (batsize, dims)
        remb = self.W[rinp, :] # (batsize, dims)
        iseq = T.stack(wemb, remb) # (2, batsize, dims)?
        iseq = iseq.dimshuffle(1, 0, 2) # (batsize, 2, dims)
        oseq = rnnu(iseq)
        om = oseq[:, np.int32(-1), :]
        return om

    @property
    def depparameters(self):
        return self.rnnu.getparams()

    @property
    def ownparameters(self):
        return [self.W]

    def gettrainf(self, inps, outps, tCost):
        # get gradients
        params = self.parameters
        grads = map(lambda x: T.grad(tCost, x).astype("float32"), params)
        updates = map(lambda (p, g): (p, (p - self.lr * self.numbats * g).astype("float32")), zip(params, grads))
        trainf = theano.function(
            inputs=inps,
            outputs=outps,
            updates=updates,
            profile=self._profiletheano
        )
        return trainf

    def getpredictfunction(self):
        prf = self.getpreddotf()

        def pf(x, y, z):
            args = [np.asarray(i).reshape((1,)).astype("int32") for i in [x, y, z]]
            return prf(*args)[0][0]

        return pf

    def getpredf(self):             # function to compute the predicted vector given entity and relation
        winp, rinp = T.ivectors("winpp", "rinpp")
        om = self.prebuilddot(winp, rinp, self.rnnu)
        return theano.function(inputs=[rinp, winp], outputs=[om])

    def getpreddotf(self):          # function to compute the score for a triple (array) given the indexes
        winp, rinp, hinp = T.ivectors("winppp", "rinppp", "hinppp")
        om = self.builddot(winp, rinp, hinp, self.rnnu)
        return theano.function(inputs=[rinp, winp,   hinp], outputs=[om])

    def getnormf(self):
        if self._normalize is True:
            norms = self.W.norm(2, axis=1).reshape((self.W.shape[0], 1))
            upd = (self.W, self.W/norms)
            return theano.function(inputs=[], outputs=[], updates=[upd])
        else:
            return None

    def getsamplegen(self, X, batsize):
        '''
        get sample generator
        :param X: indexes of nonzeroes of original input tensor. X is a ([int*]*)
        :param batsize: size of batch (number of samples generated)
        :return:
        '''
        negrate = self.negrate
        dims = X.shape
        print(dims)
        corruptrange = []
        corruptionmap = {
            "full": [0,1,2],
            "nlhs": [0,2],
            "rhs":  [2],
            "nmhs": [1,2],
            "nrhs": [0,1],
            "mhs":  [0],
            "lhs":  [1]
        }
        corruptrange = corruptionmap[self.corruption]
        xkeys = X.keys
        zvals = list(set(xkeys[:, 0]))
        print("corruptrange: ", corruptrange)

        def samplegen(): # vanilla samplegen
            corruptaxis = np.random.choice(corruptrange) # random axis to corrupt from corruptrange
            nonzeroidx = sorted(np.random.randint(0, len(X), (batsize,)).astype("int32"))
            possamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
            negsamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
            corrupted = np.random.randint(0, dims[corruptaxis], (batsize,)).astype("int32")
            negsamples[corruptaxis] = corrupted
            return possamples + negsamples
        return samplegen


class RNNTFSGDSM(Embedder, AdaDelta, Saveable, Trainable, Predictor, Batched):
    # DONE: implement Adadelta: http://arxiv.org/pdf/1212.5701.pdf ?
    #       probably after-epoch normalization breaks Adadelta
    #       ==> remove norm, use wreg, exact norm doesn't matter much for prediction anyway
    #       TODO: check implementation
    # TODO: hierarchical softmax
    # TODO: think about XE
    # TODO: validation
    def __init__(self, rnnuc=GRU, **kw):
        super(RNNTFSGDSM, self).__init__(**kw)
        self.rnnuc = rnnuc
        self.rnnu = self.rnnuc(dim=self.dims, indim=self.dims, wreg=self.wreg)
        scale = 0.1
        offset = 0.5

    def train(self, X, evalinter=10): # X: z, x, y, v OR r, s, o, v
        batsize = self.getbatsize(X)
        probs, inps, gold = self.defmodel()
        tErr = self.geterr(probs, gold)
        tReg = self.getreg()
        tCost = tErr + tReg
        trainf = self.gettrainf(inps+[gold], [tErr, tCost], tCost)
        err = self.trainloop(X=X,
                             trainf=self.getbatchloop(trainf, self.getsamplegen(X, batsize)),
                             evalinter=evalinter,
                             normf=self.getnormf())
        return err

    def defmodel(self):
        sidx, ridx, oidx = T.ivectors("sidx", "ridx", "oidx")
        outp = self.builddot(sidx, ridx, self.rnnu)
        self.smlayer = theano.shared(np.random.random((self.dims, self.vocabsize)).astype("float32")*scale-offset)
        probs = T.nnet.softmax(T.dot(outp, self.smlayer))
        #embed()
        return probs, [sidx, ridx], oidx

    def builddot(self, sidx, ridx, rnnu):
        semb = self.embed(sidx) # (batsize, dims)
        remb = self.embed(ridx) # (batsize, dims)
        iseq = T.stack(semb, remb) # (2, batsize, dims)
        iseq = iseq.dimshuffle(1, 0, 2) # (batsize, 2, dims)
        oseq = rnnu(iseq)
        om = oseq[:, np.int32(-1), :] # om is (batsize, dims)
        return om

    def getsamplegen(self, X, batsize):
        xkeys = X.keys
        def samplegen():
            nonzeroidx = sorted(np.random.randint(0, len(X), (batsize,)).astype("int32"))
            samples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
            return [samples[1], samples[0], samples[2]]     # [[s*], [r*], [o*]]
        return samplegen

    def geterr(self, x, y):
        return T.sum(T.nnet.categorical_crossentropy(x, y))

    @property
    def ownparameters(self):
        return [self.W, self.smlayer]

    @property
    def depparameters(self):
        return self.rnnu.getparams()

    def getpredictfunction(self):
        probs, inps, gold = self.defmodel()
        score = probs.T[gold]
        scoref = theano.function(inputs=inps+[gold], outputs=score)
        def pref(s, r, o):
            args = [np.asarray(i).reshape((1,)).astype("int32") for i in [s, r, o]]
            return scoref(*args)
        return pref



if __name__ == "__main__":
    m = RNNTFSGDC.load("../models/RNNTFSGDSM.2015-12-10=22:39.auto")
    #embed()
    print m.predict(11329, 9325, 7156)
