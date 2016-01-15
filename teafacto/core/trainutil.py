import cPickle as pickle
from datetime import datetime as dt
from math import ceil
import os
import sys

import numpy as np
import theano
from theano import tensor as T

from teafacto.core.optimizers import SGD, Optimizer
from teafacto.core.utils import ticktock as tt

__author__ = 'denis'


def showgraph(var):
    theano.printing.pydotprint(var, outfile="/home/denis/logreg_pydotprint_prediction.png", var_with_name_simple=True)


class Saveable(object):
    def __init__(self, autosave=False, **kw):
        super(Saveable, self).__init__(**kw)
        self._autosave = autosave
        self._autosave_filepath = None
    ############# Saving and Loading #################"
    def getsavepath(self):
        dfile = os.path.join(os.path.dirname(__file__), "../../models/%s.%s" %
                             (self.printname, dt.now().strftime("%Y-%m-%d=%H:%M")))
        return dfile

    @property
    def printname(self):
        return self.__class__.__name__

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


class Parameterized(object):
    @property
    def parameters(self):
        return self.depparameters + self.ownparameters

    @property
    def ownparameters(self):
        return []

    @property
    def depparameters(self):
        return []


class Profileable(object):
    def __init__(self, **kw):
        super(Profileable, self).__init__(**kw)
        self._profiletheano = False
    ############## PROFILING #######################
    @property
    def profiletheano(self):
        self._profiletheano = True
        return self


class SGDBase(Parameterized, Profileable):
    def __init__(self, maxiter=50, lr=0.0001, numbats=100, wreg=0.00001, **kw):
        self.tt = tt(self.__class__.__name__)
        self.maxiter = maxiter
        self.currentiter = 0
        self.numbats = numbats
        self.wreg = wreg
        self.tnumbats = theano.shared(np.float32(self.numbats), name="numbats")
        self.twreg = theano.shared(np.float32(self.wreg), name="wreg")
        self._optimizer = SGD(lr)
        super(SGDBase, self).__init__(**kw)
        self.tt.tock("initialized").tick()

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
        params = self.parameters
        grads = T.grad(cost, wrt=params)
        updates = self.getupdates(params, grads)
        #showgraph(updates[0][1])
        ret = theano.function(inputs=finps,
                               outputs=fouts,
                               updates=updates,
                               profile=self._profiletheano)
        self.tt.tock("compiled").tick()
        return ret

    def getvalidf(self, finps, fouts):
        ret = theano.function(inputs=finps, outputs=fouts)
        return ret

    def getvalidation(self, validf, samplegen):
        def validator():
            samples = samplegen()
            return validf(*samples)[0], samples[0].shape[0] # tErr
        return validator

    def getupdates(self, params, grads):
        return self._optimizer.getupdates(params, grads)

    def train(self, traindata, trainlabels, validdata=None, validlabels=None, evalinter=10): # X: z, x, y, v OR r, s, o, v
        self.batsize = int(ceil(traindata.shape[0]*1./self.numbats))
        self.tbatsize = theano.shared(np.int32(self.batsize))
        inps, tErr, tCost = self.defproblem()

        trainf = self.gettrainf(inps, [tErr, tCost], tCost)
        validf = self.getvalidf(inps, [tErr])
        if validdata is None or validlabels is None:
            validator = None
        else:
            validator = self.getvalidation(validf, self.getsamplegen(validdata, validlabels, onebatch=True))
        err, verr = self.trainloop(trainf=self.getbatchloop(trainf, self.getsamplegen(traindata, trainlabels)),
                             evalinter=evalinter,
                             normf=self.getnormf(),
                             validf=validator)
        return err, verr

    def trainloop(self, trainf, validf=None, evalinter=1, normf=None, average_err=True):
        self.tt.tick("training")
        err = []
        verr = []
        stop = False
        self.currentiter = 1
        evalcount = evalinter
        if normf:
            normf()
        while not stop:
            print("iter %d/%.0f" % (self.currentiter, float(self.maxiter)))
            start = dt.now()
            erre, tsize = trainf()
            if average_err:
                erre /= tsize
            if normf:
                normf()
            if self.currentiter == self.maxiter:
                stop = True
            self.currentiter += 1
            err.append(erre)
            if validf is not None and self.currentiter % evalinter == 0: # validate and print
                verre, vsize = validf()
                if average_err:
                    verre /= vsize
                verr.append(verre)
                print "training error: %f \t validation error: %f" % (erre, verre)
            else:
                print "training error: %f" % erre
            print("iter done in %f seconds" % (dt.now() - start).total_seconds())
            evalcount += 1
            if self._autosave:
                self.save()
        self.tt.tock("trained").tick()
        return err, verr

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
            tsize = 0
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
                tsize += sampleinps[0].shape[0]
                c += 1
            return terr, tsize
        return batchloop


class SMBase(SGDBase):
    #region SMBase
    def __init__(self, **kw):
        super(SMBase, self).__init__(**kw)

    def defproblem(self):
        probs, gold, inps = self.defmodel()
        tReg = self.getreg()
        tErr = self.geterr(probs, gold)
        tCost = tReg + tErr
        return inps, tErr, tCost

    def defmodel(self):
        raise NotImplementedError("use subclass")

    def getreg(self, regf=lambda x: T.sum(x**2), factor=1./2):
        return factor * reduce(lambda x, y: x + y,
                               map(lambda x: regf(x) * self.twreg,
                                   self.parameters), 0)

    def geterr(self, probs, gold): # cross-entropy
        return -T.mean(T.log(probs[T.arange(self.batsize), gold]))

    def getnormf(self):
        return None
    #endregion SMBase


class Normalizable(Parameterized):
    def __init__(self, normalize=False, **kw):
        super(Normalizable, self).__init__(**kw)
        self._normalize = normalize

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
        grads = T.grad(cost=tCost, wrt=params)
        updates = self.getupdates(params, grads)
        showgraph(updates[0][1])
        trainf = theano.function(
            inputs=inps,
            outputs=outps,
            updates=updates,
            profile=self._profiletheano
        )
        return trainf

    def getupdates(self, params, grads):
        return map(lambda (p, g): (p, (p - np.asarray([self.lr*self.numbats]).astype("float32") * g).astype("float32")), zip(params, grads))


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
        self.W = theano.shared((np.random.random((self.vocabsize, self.dims)).astype("float32")-offset)*scaler, name="W")

    @property
    def ownparameters(self):
        return [self.W]

    def getownreg(self):
        return (1./2.) * ((T.sum(self.W**2) * self.wreg))

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


def uniform(shape, offset=0.5, scale=1.):
    return (np.random.random(shape).astype("float32")-offset)*scale