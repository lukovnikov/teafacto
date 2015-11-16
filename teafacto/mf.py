from __future__ import print_function
import theano, time
from theano import tensor as T
import numpy as np, pandas as pd
from math import ceil, floor
from IPython import embed


class MF(object):
    '''
    abstract class for matrix factorization
    '''

    def initvars(self, X, numrows=None, numcols=None):
        '''
        initializes weight variables W and H, loads the input matrix X into a shared variable
        '''
        # initialize W and H
        numrows = X.shape[0] if numrows is None else numrows
        numcols = X.shape[1] if numcols is None else numcols
        self.numrows = numrows
        self.numcols = numcols
        self.W = theano.shared(np.random.random((numrows, self.dims)))
        self.H = theano.shared(np.random.random((self.dims, numcols)))
        # initialize data
        self.X = theano.shared(X)

    def trainloop(self, X, trainf, normf=None, evalinter=1):
        '''
        training loop that uses the trainf training function on the given data X
        '''
        err = []
        stop = False
        itercount = 0
        evalcount = evalinter
        if normf:
            normf()
        while not stop:
            print("iter %d/%d" % (itercount, self.maxiter))
            erre = trainf()
            if normf:
                normf()
            if itercount == self.maxiter:
                stop = True
            itercount += 1
            if erre is None and (evalinter != 0 and evalinter != np.infty) and evalcount == evalinter:
                error = np.linalg.norm(X - np.dot(self.W.get_value(), self.H.get_value()))
                err.append(error)
                print(error)
                evalcount = 0
            else:
                err.append(erre)
                print(erre)
            evalcount += 1
        return err


class NMFGD(MF):
    '''
    Non-negative matrix factorization with Gradient Descent
    '''

    def __init__(self, dims, Lreg=0, Rreg=0, maxiter=200, lr=0.00001):
        '''
        save parameters
        :param dims: dimension of embeddings
        :param Lreg: regularization parameter for left matrix (W)
        :param Rreg: regularization parameter for right matrix (H)
        :param maxiter: maximum number of epochs
        :param lr: learning reate
        :return:
        '''
        self.Wreg = Lreg
        self.Hreg = Rreg
        self.maxiter = maxiter
        self.dims = dims
        self.lr = lr

    def train(self, X, evalinter=10):
        '''
        function to call to train this NMF GD on given matrix X
        Calls trainingloop()
        '''
        self.initvars(X)
        # define errors and cost
        tErr = (1./2.) * ((self.X - T.dot(self.W, self.H))**2).sum()
        tReg = (1./2.) * ((self.W**2).sum() * self.Wreg + (self.H**2).sum() * self.Hreg)
        tCost = tErr + tReg
        # get gradients
        gW, gH = T.grad(tCost, [self.W, self.H])
        # define updates and function
        updW = (self.W, T.clip(self.W - self.lr * gW, 0, np.infty))
        updH = (self.H, T.clip(self.H - self.lr * gH, 0, np.infty))
        trainf = theano.function(
            inputs=[],
            outputs=[tErr],
            updates=[updW, updH]
        )
        normf = theano.function(
            inputs=[],
            outputs=[],
            updates=[
                (self.W, (self.W.T/T.sum(self.W, axis=1)).T),
                #
            ]
        )
        # train loop
        err = self.trainloop(X, trainf=trainf, evalinter=evalinter)
        return self.W.get_value(), self.H.get_value(), err


class NMFSGD(NMFGD):
    '''
    NMF with SGD
    '''
    def __init__(self, dims, numbats = 1000, Lreg=0, Rreg=0, maxiter=200, lr=0.00001):
        '''
        create new NMFSGD object and save parameters
        :param dims: number of dimensions for the embedding
        :param numbats: number of batches per epoch
        :param Lreg: regularization parameter for left matrix (W)
        :param Rreg: regularization parameter for right matrix (H)
        :param maxiter: maximum number of epochs
        :param lr: learning rate
        :return:
        '''
        super(NMFSGD, self).__init__(dims, Lreg, Rreg, maxiter, lr)
        self.numbats = numbats
        self.terre = 0

    def defmodel(self):
        '''
        define model
        :return: ([output variable, gold standard variable], [index variable for W, index variable for H])
        '''
        winp, hinp = T.ivectors("winp", "hinp")
        outp = self.X[winp, hinp]
        dotp = T.sum(self.W[winp, :] * self.H[:, hinp].T, axis=1)
        return [dotp, outp], [winp, hinp]

    def geterr(self, dotp, outp):
        '''
        return error variable for given output and gold standard variables
        here: sum of squared errors
        '''
        return (1./2.) * T.sum((outp - dotp)**2)

    def getreg(self, winp, hinp):
        '''
        return regularization variable for given input index variables
        here: l2 norm
        '''
        tReg = (1./2.) * (T.sum(self.W[winp, :]**2) * self.Wreg + T.sum(self.H[:, hinp]**2) * self.Hreg)
        #tReg = (1./2.) * (T.sum(self.W**2) * self.Wreg + T.sum(self.H**2) * self.Hreg)
        return tReg

    def gettrainf(self, inps, outps, tCost):
        '''
        get theano training function that takes inps as input variables, returns outps as outputs
        and takes the gradient of tCost w.r.t. the matrix decomposition components W and H
        :param inps:
        :param outps:
        :param tCost:
        :return:
        '''
        # get gradients
        gW = T.grad(tCost, self.W)
        gH = T.grad(tCost, self.H)

        # define updates and function
        updW = (self.W, T.clip(self.W - self.lr * self.numbats * gW, 0, np.infty))
        updH = (self.H, T.clip(self.H - self.lr * self.numbats * gH, 0, np.infty))
        trainf = theano.function(
            inputs=inps,
            outputs=outps,
            updates=[updW, updH],
            profile=True
        )
        return trainf

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
                    print("iter progress %.0f" % perc + "% ", end='\r')
                    prevperc = perc
                #endregion
                sampleinps = samplegen()
                terr += trainf(*sampleinps)[0]
                c += 1
            return terr
        return batchloop

    def getsamplegen(self, X, batsize):
        '''
        returns sample generator for given X and batsize
        '''
        def samplegen():
            # select random
            row = np.random.randint(0, X.shape[0], (batsize,)).astype("int32")
            col = np.random.randint(0, X.shape[1], (batsize,)).astype("int32")
            return [row, col]
        return samplegen

    def getbatsize(self, X):
        '''
        returns batch size (batsize)
        '''
        numsam = X.shape[0] * X.shape[1]
        batsize = ceil(numsam*1./self.numbats)
        return batsize

    def transformX(self, X):
        '''
        transform the input matrix X into some other form
        here: nothing happens, original X is returned
        '''
        return X

    def train(self, X, evalinter=10):
        '''
        call to train NMF with SGD on given matrix X
        '''
        self.initvars(X)
        origX = X
        X = self.transformX(X)

        outps, inps = self.defmodel()
        tErr = self.geterr(*outps)
        tReg = self.getreg(*inps)
        tCost = tErr + tReg
        trainf = self.gettrainf(inps, [tErr, tCost], tCost)

        batsize = self.getbatsize(X)

        err = self.trainloop(X, self.getbatchloop(trainf, self.getsamplegen(X, batsize)), evalinter=evalinter)

        return self.W.get_value(), self.H.get_value(), err


class NMFSGDC(NMFSGD):
    '''
    Contrastive NMF with SGD, based on a ranking objective.
    Observed values should be rated higher than unobserved ones (handling implicit feedback)
    '''
    def __init__(self, dims, numbats=1000, Lreg=0, Rreg=0, maxiter=200, lr=0.00001, negrate=1.):
        '''
        Create NMFSGDC object and save parameters
        :param dims: number of dimensions of the embedding
        :param numbats: number of batches per epoch
        :param Lreg: regularization parameter for left matrix (W)
        :param Rreg: regularization parameter for right matrix (H)
        :param maxiter: maximum number of epochs
        :param lr: learning rate
        :param negrate: fraction of negatives sampled for one positive example. If 1.0, one negative per positive
        :return:
        '''
        super(NMFSGDC, self).__init__(dims, numbats, Lreg, Rreg, maxiter, lr)
        self.negrate = negrate

    def getreg(self, *inp):
        '''
        return regularization variable for given input index variables
        here: l2 norm
        '''
        #tReg = (1./2.) * (T.sum(self.W[winp, :]**2) * self.Wreg + T.sum(self.H[:, hinp]**2) * self.Hreg)
        tReg = (1./2.) * (T.sum(self.W**2) * self.Wreg + T.sum(self.H**2) * self.Hreg)
        return tReg

    def getbatsize(self, X):
        '''
        returns batch size for a given X (batsize)
        '''
        numsam = np.count_nonzero(X)
        batsize = ceil(numsam*1./self.numbats)
        return batsize

    def defmodel(self):
        '''
        Define model
        :return: ([positive dot product, negative dot product],
        [positive left index variable, positive right index variable, negative left index var, negative right index var])
        '''
        winp, hinp = T.ivectors("winp", "hinp")
        nwinp, nhinp = T.ivectors("nwinp", "nhinp")
        dotp = T.sum(self.W[winp, :] * self.H[:, hinp].T, axis=1)
        ndotp = T.sum(self.W[nwinp, :] * self.H[:, nhinp].T, axis=1)
        dotp = dotp.reshape((dotp.shape[0], 1))
        ndotp = ndotp.reshape((ndotp.shape[0], 1))
        return [dotp, ndotp], [winp, hinp, nwinp, nhinp]

    def geterr(self, dotp, ndotp):
        '''
        Get error variable given positive dot product and negative dot product
        here: 0 if (positive dot) - 1 > (negative dot), otherwise 1 - (posdotp - negdotp)
              thus, we want the positive example to score at least 1.0 higher than the negative
        '''
        return T.sum(T.max(T.concatenate([T.zeros_like(dotp), 1 - dotp + ndotp], axis=1), axis=1))

    def transformX(self, X):
        '''
        returns indexes of nonzero elements of matrix X
        :return: ([first coordinates],[second coordinates])
        '''
        return np.nonzero(X)

    def getsamplegen(self, X, batsize):
        '''
        get sample generator
        :param X: indexes of nonzeroes of original input matrix. X is a ([int*]*)
        :param batsize: size of batch (number of samples generated)
        :return:
        '''
        negrate = self.negrate
        dims = (self.numrows, self.numcols)

        def samplegen():
            # sample positives
            nonzeroidx = np.random.randint(0, len(X[0]), (batsize,)).astype("int32")
            possamples = [X[ax][nonzeroidx].astype("int32") for ax in range(len(X))]
            # decide which part to corrupt
            corruptaxis = 1 #np.random.randint(0, len(X))
            # corrupt
            negsamples = [X[ax][nonzeroidx].astype("int32")
                            if ax is not corruptaxis
                            else np.random.randint(0, dims[corruptaxis], (batsize,)).astype("int32")
                            for ax in range(len(X))]
            return possamples + negsamples
        return samplegen


class NMFSGDCL(NMFSGDC):

    def initvars(self, X, numrows=None, numcols=None):
        '''
        initializes weight variables W and H, loads the input matrix X into a shared variable
        '''
        # initialize W and H
        numrows = X.shape[0] if numrows is None else numrows
        numcols = X.shape[1] if numcols is None else numcols
        self.numrows = numrows
        self.numcols = numcols
        self.W = theano.shared(np.random.random((numrows, self.dims))-0.5)
        self.H = theano.shared(np.random.random((self.dims, numcols))-0.5)
        # initialize data
        self.X = theano.shared(X)

    def defmodel(self):
        '''
        Define model
        :return: ([positive dot product, negative dot product],
        [positive left index variable, positive right index variable, negative left index var, negative right index var])
        '''
        winp, hinp = T.ivectors("winp", "hinp")
        nwinp, nhinp = T.ivectors("nwinp", "nhinp")
        dotp = T.nnet.sigmoid(T.sum(self.W[winp, :] * self.H[:, hinp].T, axis=1))
        ndotp = T.nnet.sigmoid(T.sum(self.W[nwinp, :] * self.H[:, nhinp].T, axis=1))
        dotp = dotp.reshape((dotp.shape[0], 1))
        ndotp = ndotp.reshape((ndotp.shape[0], 1))
        return [dotp, ndotp], [winp, hinp, nwinp, nhinp]

    def geterr(self, dotp, ndotp):
        '''
        Get error variable given positive dot product and negative dot product
        here: - positive dot + negative dot
        '''
        return T.sum(ndotp - dotp)

    def predict(self, idxs):
        '''
        :param win: vector of tuples of integer indexes for embeddings
        :return: vector of floats of predictions
        '''
        idxs = np.asarray(idxs).astype("int32")
        print([idxs[:, i] for i in range(idxs.shape[1])])
        winp, hinp = T.ivectors("winpp", "hinpp")
        dotp = T.nnet.sigmoid(T.sum(self.W[winp, :] * self.H[:, hinp].T, axis=1))
        pfun = theano.function(
            inputs=[winp, hinp],
            outputs=[dotp]
        )
        return pfun(*[idxs[:, i] for i in range(idxs.shape[1])])

    def gettrainf(self, inps, outps, tCost):
        '''
        get theano training function that takes inps as input variables, returns outps as outputs
        and takes the gradient of tCost w.r.t. the matrix decomposition components W and H
        :param inps:
        :param outps:
        :param tCost:
        :return:
        '''
        # get gradients
        gW = T.grad(tCost, self.W)
        gH = T.grad(tCost, self.H)

        # define updates and function
        updW = (self.W, self.W - self.lr * self.numbats * gW)
        updH = (self.H, self.H - self.lr * self.numbats * gH)
        trainf = theano.function(
            inputs=inps,
            outputs=outps,
            updates=[updW, updH],
            profile=True
        )
        return trainf

class NMFSGDN(NMFSGD):
    def __init__(self, dims, numbats = 1000, Lreg=0, Rreg=0, maxiter=200, lr=0.00001, negrate=1):
        super(NMFSGDN, self).__init__(dims, numbats, Lreg, Rreg, maxiter, lr)
        self.negrate = negrate

    def train(self, X, numrows=None, numcols=None, evalinter=10):
        self.initvars(X, numrows=numrows, numcols=numcols)
        # define errors and costs
        winp, hinp = T.ivectors("winp", "hinp")
        outp = T.fvector("outp")
        dotp = T.sum(self.W[winp, :] * self.H[:, hinp].T, axis=1)
        # embed()
        tErr = (1./2.) * T.sum((outp - dotp)**2) # MSE
        tReg = (1./2.) * (T.sum(self.W[winp, :]**2) * self.Wreg + T.sum(self.H[:, hinp]**2) * self.Hreg)
        tCost = tErr + tReg
        # embed()
        # get gradients
        gW = T.grad(tCost, self.W)
        gH = T.grad(tCost, self.H)

        numsam = X.shape[0]
        batsize = int(ceil(numsam*1./self.numbats))
        numbats = self.numbats

        # define updates and function
        updW = (self.W, T.clip(self.W - self.lr * numbats * gW, 0, np.infty))
        updH = (self.H, T.clip(self.H - self.lr * numbats * gH, 0, np.infty))
        trainf = theano.function(
            inputs=[winp, hinp, outp],
            outputs=[tErr],
            updates=[updW, updH],
            profile=True
        )

        negrate = self.negrate

        def batchloop():
            c = 0
            idxs = range(X.shape[0])
            np.random.shuffle(idxs)
            prevperc = -1.
            maxc = numbats
            ts = 0.
            toterr = 0.
            while c < maxc-1:
                sliceidxs = idxs[c*batsize: min((c+1)*batsize, len(idxs))]
                possamples = X[sliceidxs]
                posouts = np.ones((possamples.shape[0],), dtype="float32")
                negsamples = []
                for i in range(possamples.shape[0]):
                    for j in range(negrate):
                        corruptdis = possamples[i, :]
                        columntocorrupt = np.random.choice(len(corruptdis))
                        corruptdis[columntocorrupt] = np.random.randint(0, numrows if columntocorrupt == 0 else numcols)
                        negsamples.append(corruptdis)
                negsamples = np.asarray(negsamples)
                negouts = np.zeros((negsamples.shape[0],), dtype="float32")
                if possamples.ndim != negsamples.ndim:
                    embed()
                samples = np.concatenate((possamples, negsamples), axis=0)
                outs = np.concatenate((posouts, negouts))
                #region Percentage counting
                perc = round(c*100./maxc)
                if perc > prevperc:
                    print("iter progress %.0f" % perc + "% ", end='\r')
                    prevperc = perc
                #endregion

                toterr += trainf(samples[:, 0].astype("int32"), samples[:, 1].astype("int32"), outs)[0]
                c += 1
            return toterr
        err = self.trainloop(X, batchloop, evalinter=0)

        return self.W.get_value(), self.H.get_value(), err

class NMFSGDNC(NMFSGDN):
    def __init__(self, *args, **kwargs):
        super(NMFSGDNC, self).__init__(*args, **kwargs)

    def train(self, X, numrows=None, numcols=None, evalinter=10):
        self.initvars(X, numrows=numrows, numcols=numcols)
        # define errors and costs
        winp, hinp = T.ivectors("winp", "hinp")
        nwinp, nhinp = T.ivectors("nwinp", "nhinp")
        dotp = T.sum(self.W[winp, :] * self.H[:, hinp].T, axis=1)
        ndotp = T.sum(self.W[nwinp, :] * self.H[:, nhinp].T, axis=1)
        dotp = dotp.reshape((dotp.shape[0], 1))
        ndotp = ndotp.reshape((ndotp.shape[0], 1))

        #embed()

        tErr = T.sum(T.max(T.concatenate([T.zeros_like(dotp), 1 - dotp + ndotp], axis=1), axis=1)) # hinge contrast
        tReg = (1./2.) * (T.sum(self.W[winp, :]**2) * self.Wreg + T.sum(self.H[:, hinp]**2) * self.Hreg)
        tCost = tErr + tReg
        #embed()
        # get gradients
        gW = T.grad(tCost, self.W)
        gH = T.grad(tCost, self.H)

        numsam = X.shape[0]
        batsize = int(ceil(numsam*1./self.numbats))
        numbats = self.numbats

        # define updates and function
        updW = (self.W, T.clip(self.W - self.lr * numbats * gW, 0, np.infty))
        updH = (self.H, T.clip(self.H - self.lr * numbats * gH, 0, np.infty))
        trainf = theano.function(
            inputs=[winp, hinp, nwinp, nhinp],
            outputs=[tErr],
            updates=[updW, updH],
            profile=True
        )

        negrate = self.negrate

        def batchloop():
            c = 0
            idxs = range(X.shape[0])
            np.random.shuffle(idxs)
            prevperc = -1.
            maxc = numbats
            ts = 0.
            toterr = 0.
            while c < maxc-1:
                sliceidxs = idxs[c*batsize: min((c+1)*batsize, len(idxs))]
                possamples = X[sliceidxs].copy()
                samples = np.concatenate([possamples]*(negrate+1))
                samples = np.concatenate([samples, samples], axis=1)
                for i in range(samples.shape[0]):
                    corruptcolumn = np.random.choice([2, 3])
                    samples[i, corruptcolumn] = np.random.randint(0, numrows if corruptcolumn == 2 else numcols)
                #region Percentage counting
                perc = round(c*100./maxc)
                if perc > prevperc:
                    print("iter progress %.0f" % perc + "% ", end='\r')
                    prevperc = perc
                #endregion

                toterr += trainf(samples[:, 0].astype("int32"), samples[:, 1].astype("int32"),
                                 samples[:, 2].astype("int32"), samples[:, 3].astype("int32"))[0]
                c += 1
            return toterr
        err = self.trainloop(X, batchloop, evalinter=0)

        return self.W.get_value(), self.H.get_value(), err

class NMFALS(MF):
    def __init__(self, dims, Lreg=0, Rreg=0, maxiter=200, eps=0.0000000001):
        self.Wreg = Lreg
        self.Hreg = Rreg
        self.maxiter = maxiter
        self.dims = dims
        self.eps = eps

    def train(self, X, evalinter=10):
        '''
        Factorizes matrix X into a product of W and H, with W: (numrowsofX x dims) and H: (dims x numcolsofX)
        Using ALS as described in Pauca et al. (2006)
        :param X: matrix
        :return: W and H
        '''
        self.initvars(X)
        # define updates and training function
        updW = (self.W, self.W * (T.dot(self.X, self.H.T) - self.Wreg * self.W)/
                (T.dot(T.dot(self.W, self.H), self.H.T) + self.eps))
        updH = (self.H, self.H * (T.dot(self.W.T, self.X) - self.Hreg * self.H)/
                (T.dot(T.dot(self.W.T, self.W), self.H) + self.eps))
        trainf = theano.function(
            inputs=[],
            outputs=[],
            updates=[updW, updH]
        )
        normf = theano.function(
            inputs=[],
            outputs=[],
            updates=[
                (self.W, (self.W.T/T.sum(self.W, axis=1)).T),
                #
            ]
        )
        # training loop
        err = self.trainloop(X, trainf=trainf, normf=normf, evalinter=evalinter)
        return self.W.get_value(), self.H.get_value(), err