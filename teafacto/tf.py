from __future__ import print_function

import theano, time, pickle, collections
import numpy as np, pandas as pd
import sys, os

from theano import tensor as T
from theano.ifelse import ifelse
from math import ceil, floor
from datetime import datetime as dt
from IPython import embed

from utils import *


class TFSGD(object):
    def __init__(self, dims=10, maxiter=50, wregs=0.0, lr=0.0000001, negrate=1, numbats=100, corruption="rhs", invZoffset=501):
        self.dims = dims
        self.maxiter = maxiter
        if issequence(wregs):
            if len(wregs) != 3:
                raise Exception("only 3D tensors are currently supported")
            else:
                self.wregs = wregs
        elif isnumber(wregs):
            self.wregs = [wregs]*3
        else:
            raise Exception("wrong type for regularization weights")
        self.lr = lr
        self.negrate = negrate
        self.numbats = numbats
        self.corruption = corruption
        self.invZoffset = invZoffset

        self._autosave = False
        self._autosave_filepath = None
        self._profiletheano = False

    @property
    def autosave(self): # for saving after each iter
        self._autosave = True
        return self

    @property
    def profiletheano(self):
        self._profiletheano = True
        return self

    def initvars(self, X, numcols=None, numrows=None, numslices=None, central=True):
        offset = 0.0
        if central is True:
            offset = 0.5
        self.numslices = X.shape[0] if numslices is None else numslices
        self.numrows = X.shape[1] if numrows is None else numrows
        self.numcols = X.shape[2] if numcols is None else numcols
        if self.numrows != self.numcols:
            pass #raise Exception("frontal slice must be square")
        self.W = theano.shared(np.random.random((self.numrows, self.dims)) - offset)
        self.R = theano.shared(np.random.random((self.numslices, self.dims, self.dims)) - offset)
        self.H = theano.shared(np.random.random((self.numcols, self.dims)) - offset)

        '''
        print("test W")
        print(self.W[0, :].eval())
        '''

        self.params = {"w": self.W, "r": self.R, "h": self.H}

        #self.X = theano.shared(X)

    def getreg(self, *inp):
        '''
        return regularization variable for given input index variables
        here: l2 norm
        '''
        tReg = (1./2.) * (T.sum(self.R**2) * self.wregs[0]
                          + T.sum(self.W**2) * self.wregs[1]
                          + T.sum(self.H**2) * self.wregs[2])
        return tReg

    def getbatsize(self, X):
        '''
        returns batch size for a given X (batsize)
        '''
        numsam = X.count_nonzeros()
        batsize = ceil(numsam*1./self.numbats)
        return batsize

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
        #    if normf:
        #        normf()
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

    def transformX(self, X):
        '''
        returns indexes of nonzero elements of 3D tensor X
        :return: ([first coordinates],[second coordinates],[third coordinates])
        '''
        return X.nonzeros(withvals=True)

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

    def defmodel(self):
        raise NotImplementedError("use a subclass of this class - this one is abstract")

    def geterr(self, x, y):
        raise NotImplementedError("use a subclass of this class - this one is abstract")

    def train(self, X, evalinter=10):
        self.initvars(X)

        outps, inps = self.defmodel()
        tErr = self.geterr(*outps)
        tReg = self.getreg(*inps)
        tCost = tErr + tReg
        trainf = self.gettrainf(inps, [tErr, tCost], tCost)

        batsize = self.getbatsize(X)
        err = [0.]

        err = self.trainloop(X, self.getbatchloop(trainf, self.getsamplegen(X, batsize)), evalinter=evalinter)

        return self.params.values(), err

    def getbatchloop(self, trainf, samplegen):
        '''
        returns the batch loop, loaded with the provided trainf training function and samplegen sample generator
        '''
        numbats = self.numbats

        def batchloop():
            '''
            called on every new batch
            '''
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

    def gettrainf(self, inps, outps, tCost):
        '''
        get theano training function that takes inps as input variables, returns outps as outputs
        and takes the gradient of tCost w.r.t. the tensor decomposition components W, R and H
        :param inps:
        :param outps:
        :param tCost:
        :return:
        '''
        # get gradients
        gW = T.grad(tCost, self.W)
        gR = T.grad(tCost, self.R)
        gH = T.grad(tCost, self.H)

        # define updates and function
        updW = (self.W, self.W - self.lr * self.numbats * gW)
        updR = (self.R, self.R - self.lr * self.numbats * gR)
        updH = (self.H, self.H - self.lr * self.numbats * gH)
        trainf = theano.function(
            inputs=inps,
            outputs=outps,
            updates=[updW, updR, updH],
            profile=self._profiletheano
        )
        return trainf

    def save(self, filepath=None, extra=None):
        if self._autosave_filepath is not None:
            filepath = self._autosave_filepath
        if filepath is None:
            self._autosave_filepath = self.getsavepath()+".auto"
            filepath = self._autosave_filepath
        with open(filepath, "w") as f:
            pickle.dump((self.W.get_value(), self.R.get_value(), self.H.get_value(), self.getmodelparams(), extra), f)

    def getmodelparams(self):
        return   {"dims":       self.dims,
                  "maxiter":    self.maxiter,
                  "lr":         self.lr,
                  "numbats":    self.numbats,
                  "wregs":      self.wregs,
                  "negrate":    self.negrate,
                  "corruption": self.corruption,
                  "invZoffset": self.invZoffset}

    def getsavepath(self):
        dfile = os.path.join(os.path.dirname(__file__), "../models/%s.%s" %
                             (os.path.splitext(self.__class__.__name__)[0], dt.now().strftime("%Y-%m-%d=%H:%M")))
        return dfile

    @classmethod
    def load(cls, filepath):
        ret = None
        with open(filepath) as f:
            W, R, H, settings, extra = pickle.load(f)
            ret = cls(**settings)
            ret.W = theano.shared(W)
            ret.R = theano.shared(R)
            ret.H = theano.shared(H)
            ret.extra = extra
        return ret

    def embedX(self, idx):
        return self.W.get_value()[idx, :]

    def normX(self, idx):
        return np.linalg.norm(self.embedX(idx))

    def embedY(self, idx):
        return self.H.get_value()[idx, :]

    def normY(self, idx):
        return np.linalg.norm(self.embedY(idx))

    def embedZ(self, idx):
        return self.R.get_value()[idx, :, :]

    def normZ(self, idx):
        return np.linalg.norm(self.embedZ(idx))

    def embedXYdot(self, iA, iB):
        return np.dot(self.embedX(iA), self.embedY(iB))

    def embedXXdot(self, iA, iB):
        return np.dot(self.embedX(iA), self.embedX(iB))

    def embedXXcos(self, iA, iB):
        va = self.embedX(iA)
        vb = self.embedX(iB)
        return np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))

    def embedYYdot(self, iA, iB):
        return np.dot(self.embedY(iA), self.embedY(iB))

    def embedYYcos(self, iA, iB):
        va = self.embedY(iA)
        vb = self.embedY(iB)
        return np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))

    def embedXYcos(self, iA, iB):
        va = self.embedX(iA)
        vb = self.embedY(iB)
        return np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))

    def embedXYZdot(self, iT, iA, iB):
        t = self.embedZ(iT)
        a = self.embedX(iA)
        b = self.embedY(iB)
        at = np.dot(a, t)
        atb = np.dot(at, b)
        return atb

    def embedXYZcos(self, iT, iA, iB):
        t = self.embedZ(iT)
        a = self.embedX(iA)
        b = self.embedY(iB)
        at = np.dot(a, t)
        atb = np.dot(at, b) / (np.linalg.norm(at) * np.linalg.norm(b))
        return atb


class TFSGDC(TFSGD):

    def builddot(self, winp, rinp, hinp):
        wrhdot = self.builddotwos(winp, rinp, hinp)
        wrpdot = T.nnet.sigmoid(wrhdot)
        return wrpdot

    def builddotwos(self, winp, rinp, hinp):
        wemb = self.W[winp, :]
        #remb = ifelse(T.lt(rinp, self.invZoffset), self.R[rinp, :, :], T.nlinalg.matrix_inverse(self.R[rinp-self.invZoffset, :, :]))
        remb = self.R[rinp, :, :]
        hemb = self.H[hinp, :]
        wrprod = T.batched_dot(wemb, remb)
        wrhdot = T.sum(wrprod * hemb, axis=1)
        return wrhdot

    def defmodel(self):
        '''
        Define model
        '''
        winp, rinp, hinp = T.ivectors("winp", "rinp", "hinp")
        nwinp, nrinp, nhinp = T.ivectors("nwinp", "nrinp", "nhinp")
        dotp = self.builddot(winp, rinp, hinp)
        ndotp = self.builddot(nwinp, nrinp, nhinp)
        dotp = dotp.reshape((dotp.shape[0], 1))
        ndotp = ndotp.reshape((ndotp.shape[0], 1))
        return [dotp, ndotp], [rinp, winp, hinp, nrinp, nwinp, nhinp]

    def getmodelparams(self):
        return   {"dims":       self.dims,
                  "maxiter":    self.maxiter,
                  "lr":         self.lr,
                  "numbats":    self.numbats,
                  "wregs":      self.wregs,
                  "negrate":    self.negrate,
                  "corruption": self.corruption};

    def geterr(self, dotp, ndotp):
        '''
        Get error variable given positive dot product and negative dot product
        here: - positive dot + negative dot
        '''
        return T.sum(ndotp - dotp)

    def getpredf(self):
        winp, rinp, hinp = T.ivectors("winpp", "rinpp", "hinpp")
        dotp = self.builddot(winp, rinp, hinp)
        pfun = theano.function(
            inputs=[winp, rinp, hinp],
            outputs=[dotp]
        )
        return pfun

    def getpredfdot(self):
        winp, rinp, hinp = T.ivectors("winpp", "rinpp", "hinpp")
        dotp = self.builddotwos(winp, rinp, hinp)
        pfun = theano.function(
            inputs=[winp, rinp, hinp],
            outputs=[dotp]
        )
        return pfun

    def predict(self, idxs):
        '''
        :param win: vector of tuples of integer indexes for embeddings
        :return: vector of floats of predictions
        '''
        idxs = np.asarray(idxs).astype("int32")
        pfun = self.getpredf()
        return pfun(*[idxs[:, i] for i in range(idxs.shape[1])])

    def predictdot(self, idxs):
        idxs = np.asarray(idxs).astype("int32")
        pfun = self.getpredfdot()
        return pfun(*[idxs[:, i] for i in range(idxs.shape[1])])


class TFMF0SGDC(TFSGDC):
    def __init__(self, lr2=0.00000001, **kwargs):
        super(TFMF0SGDC, self).__init__(**kwargs)
        self.lr2 = lr2

    def builddot2(self, winp, crinp):
        wemb = self.W[winp, :]
        cemb = self.H[crinp, :]
        d = T.batched_dot(wemb, cemb)
        return T.nnet.sigmoid(d)

    def defmodel2(self):
        winp, crinp = T.ivectors("winp", "crinp")
        nwinp, ncrinp = T.ivectors("nwinp", "ncrinp")
        dotp = self.builddot2(winp, crinp)
        ndotp = self.builddot2(nwinp, ncrinp)
        dotp = dotp.reshape((dotp.shape[0], 1))
        ndotp = ndotp.reshape((ndotp.shape[0], 1))
        return [dotp, ndotp], [winp, crinp, nwinp, ncrinp]

    def train(self, X, evalinter=10):
        '''
        call to train NMF with SGD on given matrix X
        '''
        X.sortby(dim=1)
        self.initvars(X)
        #self.origX = X
        #X = self.transformX(X)

        outps, inps = self.defmodel()
        outps2, inps2 = self.defmodel2()
        tErr = self.geterr(*outps)
        tErr2 = self.geterr(*outps2)
        tReg = self.getreg(*inps)
        tReg2 = self.getreg(*inps2)
        tCost = tErr + tReg
        tCost2 = tErr2 + tReg2
        trainf = self.gettrainf(inps, [tErr, tCost], tCost)
        trainf2 = self.gettrainf2(inps2, [tErr2, tCost2], tCost2)

        def fulltrainf(*sampleinps):
            # sampleinps:   0 - true relt, 1 - true entity, 2 - true relc, 3 - neg relt, 4 - neg entity, 5 - neg relc
            #               6 - true relt-c, 7 - true entity, 8 - neg relt-c, 9 - neg entity
            a = trainf(*sampleinps[:6])[0]
            b = trainf2(*[sampleinps[7], sampleinps[6], sampleinps[9], sampleinps[8]])[0]
            #b2 = trainf2(*[sampleinps[7], sampleinps[6], sampleinps[9], sampleinps[6], sampleinps[9]])
            return [a + b]

        batsize = self.getbatsize(X)
        err = [0.]

        err = self.trainloop(X, self.getbatchloop(fulltrainf, self.getsamplegen(X, batsize)), evalinter=evalinter)

        return self.W.get_value(), self.R.get_value(), self.H.get_value(), err

    def gettrainf2(self, inps, outps, tCost):
        '''
        get theano training function that takes inps as input variables, returns outps as outputs
        and takes the gradient of tCost w.r.t. the tensor decomposition components W, R and H
        :param inps:
        :param outps:
        :param tCost:
        :return:
        '''
        # get gradients
        gW = T.grad(tCost, self.W)
        gH = T.grad(tCost, self.H)

        # define updates and function
        updW = (self.W, self.W - self.lr2 * self.numbats * gW)
        updH = (self.H, self.H - self.lr2 * self.numbats * gH)
        trainf = theano.function(
            inputs=inps,
            outputs=outps,
            updates=[updW, updH],
            profile=self._profiletheano
        )
        return trainf

    def getsamplegen(self, X, batsize):
        supersamplegen = super(TFMF0SGDC, self).getsamplegen(X, batsize)
        dims = X.shape
        xkeys = X.keys
        zvals = list(set(xkeys[:, 0]))
        def samplegen():
            samples = supersamplegen()
            samples.append(samples[0])
            samples.append(samples[1])
            # corrupt both
            nreltcs = np.random.choice(zvals, samples[0].shape).astype("int32")
            nents = np.random.randint(0, dims[1], samples[1].shape).astype("int32")
            # choose randomly which to corrupt
            which = np.random.choice([0, 1])
            if which == 0:
                samples.append(nreltcs)
                samples.append(samples[1])
            elif which == 1:
                samples.append(samples[0])
                samples.append(nents)
            return samples
        return samplegen


class TFMXSGDC(TFSGDC):

    def initvars(self, X, numcols=None, numrows=None, numslices=None, central=True):
        offset = 0.0
        if central is True:
            offset = 0.5
        self.numslices = X.shape[0] if numslices is None else numslices
        self.numrows = X.shape[1] if numrows is None else numrows
        self.numcols = X.shape[2] if numcols is None else numcols
        if self.numrows != self.numcols:
            pass #raise Exception("frontal slice must be square")
        self.W = theano.shared(np.random.random((self.numrows, self.dims)) - offset)
        self.R = theano.shared(np.random.random((self.numslices, self.dims, self.dims)) - offset)
        self.H = theano.shared(np.random.random((self.numslices+self.invZoffset, self.dims)) - offset)

        '''
        print("test W")
        print(self.W[0, :].eval())
        '''

        self.params = {"w": self.W, "r": self.R, "h": self.H}

    def defmodel(self):
        '''
        Define model
        '''
        def sigm(x):
            return T.nnet.sigmoid(x)
        winp, rinp, hinp, rcinp, rcinpi = T.ivectors("winp", "rinp", "hinp", "rcinp", "rcinpi")
        nwinp, nrinp, nhinp, nrcinp, nrcinpi = T.ivectors("nwinp", "nrinp", "nhinp", "nrcinp", "nrcinpi")
        dotp = self.builddotwos(winp, rinp, hinp)
        dotr = self.builddotrel(winp, rcinp)
        dotri = self.builddotrel(hinp, rcinpi)
        dotp = sigm(dotp) + sigm(dotr) + sigm(dotri)
        ndotp = self.builddotwos(nwinp, nrinp, nhinp)
        ndotr = self.builddotrel(nwinp, nrcinp)
        ndotri = self.builddotrel(nhinp, nrcinpi)
        ndotp = sigm(ndotp) + sigm(ndotr) + sigm(ndotri)
        #dotp = T.nnet.sigmoid(dotp)
        #ndotp = T.nnet.sigmoid(ndotp)
        dotp = dotp.reshape((dotp.shape[0], 1))
        ndotp = ndotp.reshape((ndotp.shape[0], 1))
        return [dotp, ndotp], [rinp, winp, hinp, rcinp, rcinpi, nrinp, nwinp, nhinp, nrcinp, nrcinpi]

    def builddotrel(self, winp, cinpi):
        wemb = self.W[winp, :]
        cemb = self.H[cinpi, :]
        return T.batched_dot(wemb, cemb)

    def builddot(self, winp, rinp, hinp):
        wrhdot = self.builddotwos(winp, rinp, hinp)
        return T.nnet.sigmoid(wrhdot)

    def builddotwos(self, winp, rinp, hinp):
        wemb = self.W[winp, :]
        #remb = ifelse(T.lt(rinp, self.invZoffset), self.R[rinp, :, :], T.nlinalg.matrix_inverse(self.R[rinp-self.invZoffset, :, :]))
        remb = self.R[rinp, :, :]
        hemb = self.W[hinp, :]
        wrprod = T.batched_dot(wemb, remb)
        wrhdot = T.sum(wrprod * hemb, axis=1)
        return wrhdot

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
        invZoffset = self.invZoffset

        def samplegen(): # vanilla samplegen
            corruptaxis = np.random.choice(corruptrange) # random axis to corrupt from corruptrange
            nonzeroidx = sorted(np.random.randint(0, len(X), (batsize,)).astype("int32"))
            possamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
            possamples.append(possamples[0])
            possamples.append(possamples[0]+invZoffset)
            negsamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
            if corruptaxis == 0:
                corrupted = np.random.choice(zvals, (batsize,)).astype("int32")
            else:
                corrupted = np.random.randint(0, dims[corruptaxis], (batsize,)).astype("int32")
            negsamples[corruptaxis] = corrupted
            negsamples.append(negsamples[0])
            negsamples.append(negsamples[0]+invZoffset)
            return possamples + negsamples
        return samplegen


class RESCALSGDC(TFSGDC):

    def initvars(self, X, numcols=None, numrows=None, numslices=None, central=True):
        offset = 0.0
        if central is True:
            offset = 0.5
        self.numslices = X.shape[0] if numslices is None else numslices
        self.numrows = X.shape[1] if numrows is None else numrows
        self.numcols = X.shape[2] if numcols is None else numcols
        if self.numrows != self.numcols:
            pass #raise Exception("frontal slice must be square")
        self.W = theano.shared(np.random.random((self.numrows, self.dims)) - offset)
        self.R = theano.shared(np.random.random((self.numslices, self.dims, self.dims)) - offset)

        '''
        print("test W")
        print(self.W[0, :].eval())
        '''

        self.params = {"w": self.W, "r": self.R}

    def getreg(self, *inp):
        '''
        return regularization variable for given input index variables
        here: l2 norm
        '''
        tReg = (1./2.) * (T.sum(self.R**2) * self.wregs[0]
                          + T.sum(self.W**2) * self.wregs[1])
        return tReg

    def defmodel(self):
        '''
        Define model
        '''
        winp, rinp, hinp = T.ivectors("winp", "rinp", "hinp")
        nwinp, nrinp, nhinp = T.ivectors("nwinp", "nrinp", "nhinp")
        dotp = self.builddot(winp, rinp, hinp)
        ndotp = self.builddot(nwinp, nrinp, nhinp)
        dotp = dotp.reshape((dotp.shape[0], 1))
        ndotp = ndotp.reshape((ndotp.shape[0], 1))
        return [dotp, ndotp], [rinp, winp, hinp, nrinp, nwinp, nhinp]

    def builddot(self, winp, rinp, hinp):
        wrhdot = self.builddotwos(winp, rinp, hinp)
        return T.nnet.sigmoid(wrhdot)

    def builddotwos(self, winp, rinp, hinp):
        wemb = self.W[winp, :]
        #remb = ifelse(T.lt(rinp, self.invZoffset), self.R[rinp, :, :], T.nlinalg.matrix_inverse(self.R[rinp-self.invZoffset, :, :]))
        remb = self.R[rinp, :, :]
        hemb = self.W[hinp, :]
        wrprod = T.batched_dot(wemb, remb)
        wrhdot = T.sum(wrprod * hemb, axis=1)
        return wrhdot

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
        invZoffset = self.invZoffset

        def samplegen(): # vanilla samplegen
            corruptaxis = np.random.choice(corruptrange) # random axis to corrupt from corruptrange
            nonzeroidx = sorted(np.random.randint(0, len(X), (batsize,)).astype("int32"))
            possamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
            negsamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
            if corruptaxis == 0:
                corrupted = np.random.choice(zvals, (batsize,)).astype("int32")
            else:
                corrupted = np.random.randint(0, dims[corruptaxis], (batsize,)).astype("int32")
            negsamples[corruptaxis] = corrupted
            return possamples + negsamples
        return samplegen

    def gettrainf(self, inps, outps, tCost):
        '''
        get theano training function that takes inps as input variables, returns outps as outputs
        and takes the gradient of tCost w.r.t. the tensor decomposition components W, R and H
        :param inps:
        :param outps:
        :param tCost:
        :return:
        '''
        # get gradients
        gW = T.grad(tCost, self.W)
        gR = T.grad(tCost, self.R)

        # define updates and function
        updW = (self.W, self.W - self.lr * self.numbats * gW)
        updR = (self.R, self.R - self.lr * self.numbats * gR)
        trainf = theano.function(
            inputs=inps,
            outputs=outps,
            updates=[updW, updR],
            profile=self._profiletheano
        )
        return trainf

    def save(self, filepath=None, extra=None):
        if self._autosave_filepath is not None:
            filepath = self._autosave_filepath
        if filepath is None:
            self._autosave_filepath = self.getsavepath()+".auto"
            filepath = self._autosave_filepath
        with open(filepath, "w") as f:
            pickle.dump((self.W.get_value(), self.R.get_value(), None, self.getmodelparams(), extra), f)