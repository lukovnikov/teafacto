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
                 batsize=20,
                 initmult=0.01,
                 gateactivation=T.nnet.sigmoid,
                 outpactivation=T.tanh):
        self.dim = dim
        self.indim = indim
        self.wreg = wreg
        self.initmult = initmult
        self.batsize = batsize
        self.gateactivation = gateactivation
        self.outpactivation = outpactivation

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
        self.initstate = T.zeros((self.batsize, self.indim))

    def rec(self, x_t, h_tm1):
        '''
        :param x_t: input values (nb_samples, nb_feats) for this recurrence step
        :param h_tm1: previous states (nb_samples, out_dim)
        :return: new state (nb_samples, out_dim)
        '''
        z = self.gateactivation(T.dot(h_tm1, self.uz) + T.dot(x_t, self.wz) + self.bz)
        r = self.gateactivation(T.dot(h_tm1, self.ur) + T.dot(x_t, self.wr) + self.br)
        hh = self.outpactivation(T.dot(h_tm1 * r, self.u) + T.dot(x_t, self.w))
        h = z * h_tm1 + (1-z) * hh
        return h

    def recur(self, x):
        inputs = x.dimshuffle(1, 0, 2)
        outputs, _ = theano.scan(fn=self.rec,
                                 sequences=inputs,
                                 outputs_info=self.initstate,
                                 n_steps=inputs.shape[0])
        return outputs.dimshuffle(1, 0, 2)

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
        self.initparams()
        return self.recur(x)

    def getreg(self):
        def regf(x):
            return T.sum(x**2)
        reg = (1./2.) * reduce(lambda x, y: x+y, map(lambda x: regf(getattr(self, x))*self.wreg, self.paramnames))
        return reg

    def getparams(self):
        return map(lambda x: getattr(self, x), self.paramnames)


class RNNTFSGDC(object):
    def __init__(self, rnnuc,
                 dims=10,
                 vocabsize=10,
                 maxiter=50,
                 wreg=0.0,
                 lr=0.0000001,
                 negrate=1,
                 numbats=100,
                 corruption="rhs"):
        self.dims = dims
        self.maxiter = maxiter
        self.vocabsize = vocabsize
        self.rnnuc = rnnuc
        self.wreg = wreg
        self.lr = lr
        self.negrate = negrate
        self.numbats = numbats
        self.corruption = corruption

        self._autosave = False
        self._autosave_filepath = None
        self._profiletheano = False

    def train(self, X, evalinter=10):
        self.batsize = self.getbatsize(X)
        outps, inps = self.defmodel()
        tErr = self.geterr(*outps)
        tReg = self.getreg(*inps)
        tCost = tErr + tReg
        trainf = self.gettrainf(inps, [tErr, tCost], tCost)
        err = self.trainloop(X, self.getbatchloop(trainf, self.getsamplegen(X, self.batsize)), evalinter=evalinter, normf=self.getnormf())
        return err

    def geterr(self, x, y):
        return T.sum(y - x)

    def getreg(self, *inp): # l2 regularization
        return (1./2.) * (T.sum(self.W**2) * self.wreg) + self.rnnu.getreg()

    def transformX(self, X):
        return X.nonzeros(withvals=True)

    def getbatsize(self, X):
        numsam = X.count_nonzeros()
        batsize = ceil(numsam*1./self.numbats)
        return batsize

    def defmodel(self):
        offset = 0.5
        self.W = theano.shared(np.random.random((self.vocabsize, self.dims)) - offset)
        self.rnnu = self.rnnuc(dim=self.dims, indim=self.dims, batsize=self.batsize, wreg=self.wreg)
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
        return omdot

    def prebuilddot(self, winp, rinp, rnnu):
        wemb = self.W[winp, :]
        remb = self.W[rinp, :]
        iseq = T.stack(wemb, remb)
        oseq = rnnu(iseq)
        om = oseq[:, -1, :]
        return om

    def getparams(self):
        return self.rnnu.getparams() + [self.W]

    def gettrainf(self, inps, outps, tCost):
        # get gradients
        params = self.getparams()
        grads = map(lambda x: T.grad(tCost, x), params)
        updates = map(lambda (p, g): (p, g - self.lr * self.numbats * g), zip(params, grads))
        trainf = theano.function(
            inputs=inps,
            outputs=outps,
            updates=updates,
            profile=self._profiletheano
        )
        return trainf

    def getpredf(self):             # function to compute the predicted vector given entity and relation
        winp, rinp = T.ivectors("winpp", "rinpp")
        om = self.prebuilddot(winp, rinp, self.rnnu)
        return theano.function(inputs=[winp, rinp], outputs=[om])

    def getpreddotf(self):          # function to compute the score for a triple (array) given the indexes
        winp, rinp, hinp = T.ivectors("winppp", "rinppp", "hinppp")
        om = self.builddot(winp, rinp, hinp, self.rnnu)
        return theano.function(inputs=[winp, rinp, hinp], outputs=[om])

    def getnormf(self):
        return None

    def trainloop(self, X, trainf, validf=None, evalinter=1, normf=None):
        '''
        training loop that uses the trainf training function on the given data X
        '''
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

    ############## PROFILING #######################
    @property
    def profiletheano(self):
        self._profiletheano = True
        return self



if __name__ == "__main__":
    m = RNNTFSGDC(GRU, 10,100,10,0.1,0.001,1,100,"rhs")
    m.train(np.random.random((10, 10)))