from __future__ import print_function
__author__ = 'denis'

''' Matrix factorization implemented in TensorFlow'''

import tensorflow as tf, numpy as np
from math import ceil


class MF(object):
    '''
    abstract class for matrix factorization
    '''

    def __init__(self, dims=10, maxiter=100):
        self.dims = dims
        self.maxiter = maxiter

    def initvars(self, X, numrows=None, numcols=None):
        '''
        initializes weight variables W and H, loads the input matrix X into a shared variable
        '''
        # initialize W and H
        numrows = X.shape[0] if numrows is None else numrows
        numcols = X.shape[1] if numcols is None else numcols
        self.numrows = numrows
        self.numcols = numcols
        self.W = tf.Variable(np.random.random((numrows, self.dims)).astype("float32"))
        self.H = tf.Variable(np.random.random((self.dims, numcols)).astype("float32"))
        # initialize data
        self.X = tf.constant(X)

    def trainloop(self, X, trainf, normf=None, evalinter=1, session=None):
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
            trainf()
            if normf:
                normf()
            if itercount == self.maxiter:
                stop = True
            erre = None
            if erre is not None and evalinter > 0 and itercount % evalinter == 0:
                erre = session.run(self.getreconstrerr(), feed_dict={self.X: X})
            itercount += 1
            err.append(erre)
            print(erre)
            evalcount += 1
        return err

    def getreconstrerr(self):
        return tf.reduce_sum(tf.pow((self.X - tf.matmul(self.W, self.H)), 2))


class NMFALS(MF):
    def __init__(self, dims, maxiter=200, Lreg=0, Rreg=0, eps=0.0000000001):
        super(NMFALS, self).__init__(dims, maxiter)
        self.Wreg = Lreg
        self.Hreg = Rreg
        self.eps = eps

    def train(self, X, evalinter=10):
        '''
        Factorizes matrix X into a product of W and H, with W: (numrowsofX x dims) and H: (dims x numcolsofX)
        Using ALS as described in Pauca et al. (2006)
        :param X: matrix
        :return: W and H
        '''
        self.initvars(X)
        W = self.W
        W = 1
        # define updates and training function
        updW = self.W.assign(self.W * (tf.matmul(self.X, tf.transpose(self.H)) - self.Wreg * self.W)/
                             (tf.matmul(tf.matmul(self.W, self.H), tf.transpose(self.H)) + self.eps))
        updH = self.H.assign(self.H * (tf.matmul(tf.transpose(self.W), self.X) - self.Hreg * self.H)/
                             (tf.matmul(tf.matmul(tf.transpose(self.W), self.W), self.H) + self.eps))
        normW = self.W.assign(self.W / tf.expand_dims(tf.reduce_sum(self.W, 0), 0))
        '''
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
        '''
        # training loop
        w = None
        h = None
        with tf.Session() as s:
            s.run(tf.initialize_all_variables())
            err = self.trainloop(X, trainf=self.dotrain([updW, updH], s), normf=self.donorm(normW, s),
                                 evalinter=evalinter, session=s)
            w = s.run(self.W)
            h = s.run(self.H)
        return w, h, err

    def dotrain(self, upds, session):
        def innertrain():
            session.run(upds)
        return innertrain

    def donorm(self, norms, session):
        def innernorm():
            session.run(norms)
        return innernorm


class NMFSGDC(MF):
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
        super(NMFSGDC, self).__init__(dims, maxiter)
        self.Wreg = Lreg
        self.Hreg = Rreg
        self.lr = lr
        self.numbats = numbats
        self.terre = 0

    def defmodel(self):
        '''
        define model
        :return: ([output variable, gold standard variable], [index variable for W, index variable for H])
        '''
        winp = tf.placeholder(dtype="int32", shape=[None], name="winp")
        hinp = tf.placeholder(dtype="int32", shape=[None], name="hinp")

        #outp = self.X[winp, hinp]  # indexing the tensor doesn't work like in numpy/Theano
        outp = tf.placeholder(dtype="float32", shape=[None], name="outp")
        # TODO this line not work
        dotp = tf.nn.sigmoid(tf.reduce_sum(tf.nn.embedding_lookup(self.W, winp)
                             * tf.nn.embedding_lookup(tf.transpose(self.H), hinp), 1))
        return winp, hinp, outp, dotp

    def geterr(self, dotp, outp):
        '''
        return error variable for given output and gold standard variables
        here: sum of squared errors
        '''
        return (1./2.) * tf.reduce_sum(tf.pow((outp - dotp), 2))

    def getreg(self, winp, hinp):
        '''
        return regularization variable for given input index variables
        here: l2 norm
        '''
        tReg = (1./2.) * (tf.reduce_sum(tf.pow(tf.nn.embedding_lookup(self.W, winp), 2)) * self.Wreg
                          + tf.reduce_sum(tf.pow(tf.nn.embedding_lookup(tf.transpose(self.H), hinp), 2)) * self.Hreg)
        #tReg = (1./2.) * (T.sum(self.W**2) * self.Wreg + T.sum(self.H**2) * self.Hreg)
        return tReg

    def gettrainf(self, tCost):
        return tf.train.GradientDescentOptimizer(self.lr).minimize(tCost)

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
                trainf(*sampleinps)
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
            outp = X[row, col]
            return [row, col, outp]
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

        winp, hinp, outp, dotp = self.defmodel()
        tErr = self.geterr(dotp, outp)
        tReg = self.getreg(winp, hinp)
        tCost = tErr + tReg
        trainf = self.gettrainf(tCost)

        batsize = self.getbatsize(X)

        w = None
        h = None
        with tf.Session() as s:
            s.run(tf.initialize_all_variables())
            err = self.trainloop(X, self.getbatchloop(self.dotrain(trainf, s, [winp, hinp, outp]),
                                                      self.getsamplegen(X, batsize)),
                                 evalinter=evalinter, session=s)
            w = s.run(self.W)
            h = s.run(self.H)
        return w, h, err

    def dotrain(self, upds, session, inps):
        def innertrain(*vals):
            d = dict(zip(inps, vals))
            session.run(upds, feed_dict=d)
            #session.run(tf.assign(self.W, tf.clip_by_value(self.W, 0., np.infty)))
            #session.run(tf.assign(self.H, tf.clip_by_value(self.H, 0., np.infty)))
        return innertrain


if __name__ == "__main__":
    mf = MF()
    mf.initvars(np.random.randint(0, 1, (10, 10)))

    nmf = NMFALS(dims=5)
    inX = np.random.randint(0, 2, (500, 1000)).astype("float32")
    nmf.train(inX)