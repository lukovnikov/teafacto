from __future__ import print_function

import theano, time, pickle, collections
import numpy as np, pandas as pd
import sys

from theano import tensor as T
from math import ceil, floor
from datetime import datetime as dt
from IPython import embed

from utils import *


class OTFSGD(object):
    '''
    only for 3D tensors of shape (slices: Nr, rows: Nx, cols: Nx)
    '''
    def __init__(self, dims=10, maxiter=50, wregs=0.0, lr=0.0000001, negrate=1, numbats=100, wsplit=0, corruption="rhs"):
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
        self.wsplit = wsplit
        self.corruption = corruption

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

        self.X = theano.shared(X)

    def getreg(self, *inp):
        '''
        return regularization variable for given input index variables
        here: l2 norm
        '''
        tReg = (1./2.) * (T.sum(self.R**2) * self.wregs[0]
                          + T.sum(self.W[0:self.wsplit, :]**2) * self.wregs[1]
                          + T.sum(self.W[self.wsplit:, :]**2) * self.wregs[2])
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
        corruption = self.corruption
        corruptrange = corruptionmap[self.corruption]
        wsplit = self.wsplit
        xkeys = X.keys
        zvals = list(set(xkeys[:, 0]))
        sep = X.firstidxover(dim=1, val=self.wsplit)
        print("corruptrange: ", corruptrange)
        relidxoffset = wsplit

        def samplegen():
            corruptaxis = np.random.choice(corruptrange)
            negsamples = []
            '''
            if corruption == "rhs" and False:
                nonzeroidx = np.random.randint(0, len(X), (batsize,)).astype("int32")
                possamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
                negsamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
                for i, x in enumerate(negsamples[1], start=0):
                    if x < wsplit:
                        corruptaxis = 2
                        corrupted = np.random.randint(wsplit, dims[corruptaxis])
                    else:
                        corruptaxis = 1
                        corrupted = np.random.randint(wsplit, dims[corruptaxis])
                    negsamples[corruptaxis][i] = corrupted
            else:
            '''
            if corruptaxis == 0:
                nonzeroidx = np.random.randint(0, len(X), (batsize,)).astype("int32")
                possamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
                negsamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
                corrupted = np.random.choice(zvals, possamples[corruptaxis].shape).astype("int32")
                negsamples[corruptaxis] = corrupted
            else:
                nonzeroidx = sorted(np.random.randint(0, len(X), (batsize,)).astype("int32"))
                possamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
                negsamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
                nonzeroidxsplitpoint = np.searchsorted(nonzeroidx, [sep])[0]
                if corruptaxis == 1: # corrupt entities
                    negsamples[1][:nonzeroidxsplitpoint] \
                        = np.random.randint(0, wsplit, (nonzeroidxsplitpoint,)).astype("int32")
                    negsamples[2][nonzeroidxsplitpoint:] \
                        = np.random.randint(0, wsplit, (batsize - nonzeroidxsplitpoint,)).astype("int32")
                elif corruptaxis == 2: # corrupt relations/rhs entities
                    negsamples[2][:nonzeroidxsplitpoint] \
                        = np.random.randint(wsplit, dims[1], (nonzeroidxsplitpoint,)).astype("int32")
                        #= np.random.randint(0, dims[1], (nonzeroidxsplitpoint,)).astype("int32")
                        #
                    negsamples[1][nonzeroidxsplitpoint:] \
                        = np.random.randint(wsplit, dims[1], (batsize - nonzeroidxsplitpoint,)).astype("int32")
            return possamples + negsamples
        return samplegen

    def defmodel(self):
        raise NotImplementedError("use a subclass of this class - this one is abstract")

    def geterr(self, x, y):
        raise NotImplementedError("use a subclass of this class - this one is abstract")

    def train(self, X, evalinter=10):
        '''
        call to train NMF with SGD on given matrix X
        '''
        X.sortby(dim=1)
        self.initvars(X)
        #self.origX = X
        #X = self.transformX(X)

        outps, inps = self.defmodel()
        tErr = self.geterr(*outps)
        tReg = self.getreg(*inps)
        tCost = tErr + tReg
        trainf = self.gettrainf(inps, [tErr, tCost], tCost)

        batsize = self.getbatsize(X)
        err = [0.]

        err = self.trainloop(X, self.getbatchloop(trainf, self.getsamplegen(X, batsize)), evalinter=evalinter)

        return self.W.get_value(), self.R.get_value(), err

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

        # define updates and function
        updW = (self.W, self.W - self.lr * self.numbats * gW)
        updR = (self.R, self.R - self.lr * self.numbats * gR)
        trainf = theano.function(
            inputs=inps,
            outputs=outps,
            updates=[updW, updR],
            profile=True
        )
        return trainf

    def save(self, filepath, extra=None):
        with open(filepath, "w") as f:
            pickle.dump((self.W.get_value(), self.R.get_value(), self.getmodelparams(), extra), f)

    @classmethod
    def load(cls, filepath):
        ret = None
        with open(filepath) as f:
            W, R, settings, extra = pickle.load(f)
            ret = cls(**settings)
            ret.W = theano.shared(W)
            ret.R = theano.shared(R)
            ret.extra = extra
        return ret

    def embedXY(self, idx):
        return self.W.get_value()[idx, :]

    def normXY(self, idx):
        return np.linalg.norm(self.embedXY(idx))

    def embedZ(self, idx):
        return self.R.get_value()[idx, :, :]

    def normZ(self, idx):
        return np.linalg.norm(self.embedZ(idx))

    def embedXYdot(self, iA, iB):
        return np.dot(self.embedXY(iA), self.embedXY(iB))

    def embedXYcos(self, iA, iB):
        va = self.embedXY(iA)
        vb = self.embedXY(iB)
        return np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))

    def embedXYZdot(self, iT, iA, iB):
        t = self.embedZ(iT)
        a = self.embedXY(iA)
        b = self.embedXY(iB)
        at = np.dot(a, t)
        atb = np.dot(at, b)
        return atb

    def embedXYZcos(self, iT, iA, iB):
        t = self.embedZ(iT)
        a = self.embedXY(iA)
        b = self.embedXY(iB)
        at = np.dot(a, t)
        atb = np.dot(at, b) / (np.linalg.norm(at) * np.linalg.norm(b))
        return atb


class OTFSGDC(OTFSGD):

    def builddot(self, winp, rinp, hinp):
        wrhdot = self.builddotwos(winp, rinp, hinp)
        wrpdot = T.nnet.sigmoid(wrhdot)
        return wrpdot

    def builddotwos(self, winp, rinp, hinp):
        wemb = self.W[winp, :]
        remb = self.R[rinp, :, :]
        hemb = self.W[hinp, :]
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
                  "wsplit":     self.wsplit,
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


class OTFMF0SGDC(OTFSGDC):
    def __init__(self, relidxoffset=0, revreloffset=0, lr2=0.00000001, **kwargs):
        super(OTFMF0SGDC, self).__init__(**kwargs)
        self.relidxoffset = relidxoffset
        self.revreloffset = revreloffset
        self.lr2 = lr2

    def builddot2(self, winp, crinp):
        wemb = self.W[winp, :]
        cemb = self.W[crinp, :]
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

        return self.W.get_value(), self.R.get_value(), err

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

        # define updates and function
        updW = (self.W, self.W - self.lr2 * self.numbats * gW)
        trainf = theano.function(
            inputs=inps,
            outputs=outps,
            updates=[updW],
            profile=True
        )
        return trainf

    def getsamplegen(self, X, batsize):
        supersamplegen = super(OTFMF0SGDC, self).getsamplegen(X, batsize)
        wsplit = self.wsplit
        relidxoffset = self.relidxoffset
        revreloffset = self.revreloffset
        dims = X.shape
        xkeys = X.keys
        zvals = list(set(xkeys[:, 0]))
        def samplegen():
            samples = supersamplegen()
            # !!! assume samples are sorted by dimension 1
            splitpoint = np.searchsorted(samples[1], [wsplit])[0]
            reltcs = samples[0][:splitpoint]+relidxoffset # only first part, forward trans-properties
            reltcs = np.concatenate((reltcs, samples[0][splitpoint:]+relidxoffset+revreloffset))
            samples.append(reltcs)
            ents = np.concatenate((samples[1][:splitpoint], samples[2][splitpoint:]))
            samples.append(ents)
            # corrupt both
            nreltcs = np.random.choice(
                map(lambda x: x+relidxoffset, zvals)
                + map(lambda x: x+relidxoffset+revreloffset, zvals)
                , reltcs.shape).astype("int32")
            nents = np.random.randint(0, wsplit, ents.shape).astype("int32")
            # choose randomly which to corrupt
            which = np.random.choice([0, 1])
            if which == 0:
                samples.append(nreltcs)
                samples.append(ents)
            elif which == 1:
                samples.append(reltcs)
                samples.append(nents)
            return samples
        return samplegen

    def getsamplegenold(self, X, batsize): #TODO
        '''
        get sample generator
        :param X: indexes of nonzeroes of original input tensor. X is a ([int*]*)
        :param batsize: size of batch (number of samples generated)
        :return:
        '''
        negrate = self.negrate
        dims = X.shape
        corruptrange = [2]
        if self.corruption == "full":
            corruptrange = [0, 1, 2]
        elif self.corruption == "nlhs":
            corruptrange = [0, 2]
        wsplit = self.wsplit
        xkeys = X.keys
        zvals = list(set(xkeys[:, 0]))
        relidxoffset = self.relidxoffset

        def samplegen():
            # decide which part to corrupt
            corruptaxis = 2 if len(corruptrange) < 2 else np.random.choice(corruptrange)
            # corrupt
            if len(corruptrange) > 1:
                # chose to corrupt before or after wsplit
                '''corruptbefore = np.random.random() > wsplit*1.0/dims[1]
                if corruptbefore:
                    nonzeroidx = np.random.randint(0, wsplit, (batsize,)).astype("int32")
                else:
                    nonzeroidx = np.random.randint(wsplit, len(X)).astype("int32")
                possamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
                negsamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
                if corruptbefore:
                    corrupted = np.random.randint()'''
                nonzeroidx = np.random.randint(0, len(X), (batsize,)).astype("int32")
                possamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
                negsamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
                for i, x in enumerate(negsamples[0], start=0):
                    if corruptaxis == 0: # corrupting z-axis
                        corrupted = np.random.choice(zvals)
                    elif corruptaxis == 1 or corruptaxis == 2: # corrupting x-axis
                        v = negsamples[corruptaxis]
                        if v < wsplit:
                            corrupted = np.random.randint(0, min(wsplit, dims[corruptaxis]))
                        else:
                            corrupted = np.random.randint(wsplit, dims[corruptaxis])
                    negsamples[corruptaxis][i] = corrupted
            else: # for every negatively sampled RHS, also neg-sample a MHS-C based on MHS but beware: MHS uses different indexes than MHS-C ==> need to translate
                # sample positives
                nonzeroidx = np.random.randint(0, len(X), (batsize,)).astype("int32")
                possamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
                possamples.append(possamples[0]+relidxoffset) # transform idx's from MHS to MHS-C
                negsamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
                negsamples.append(np.random.choice(zvals, negsamples[0].shape).astype("int32")+relidxoffset) # add corruption to MHS-C
                for i, x in enumerate(negsamples[1], start=0):
                    if x < wsplit:
                        corruptaxis = 2
                        if negsamples[2][i] < wsplit:
                            corrupted = np.random.randint(0, min(wsplit, dims[corruptaxis]))
                        else:
                            corrupted = np.random.randint(wsplit, dims[corruptaxis])
                    else:
                        corruptaxis = 1
                        corrupted = np.random.randint(wsplit, dims[corruptaxis])
                    negsamples[corruptaxis][i] = corrupted
            return possamples + negsamples
        return samplegen