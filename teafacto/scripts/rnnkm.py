__author__ = 'denis'

from datetime import datetime
import pickle
import os

from matplotlib import pyplot as plt
import numpy as np

from teafacto.core.sptensor import SparseTensor
from teafacto.core.utils import ticktock as TT




# from teafacto.kmsm import RNNEKMSM, AutoRNNEKMSM, AutoRNNEKMSM
from teafacto.smsm import RNNESMSM
from teafacto.kmm import AddEKMM, VecMulEKMM, MatMulEKMM, RNNEKMM, RNNEOKMM, VecMulEKMMDist, TransAddEKMM

from teafacto.core.optimizers import SGD
from teafacto.core.rnn import GRU, IFGRU, LSTM, IFGRUTM

from teafacto.kmm import EKMM
import theano
from theano import tensor as T

np.random.seed(12345)


def loaddata(file):
    file = os.path.join(os.path.dirname(__file__), file)
    st = SparseTensor.from_ssd(file)
    return st

def loaddic(file):
    file = os.path.join(os.path.dirname(__file__), file)
    return pickle.load(open(file))

def loadmeta(dfile):
    dfile = os.path.join(os.path.dirname(__file__), dfile)
    meta = pickle.load(open(dfile))
    return meta

def getsavepath():
    dfile = os.path.join(os.path.dirname(__file__), "../../models/%s.%s" %
                         (os.path.splitext(os.path.basename(__file__))[0], datetime.now().strftime("%Y-%m-%d=%H:%M")))
    return dfile

def run():
    # params
    dims = 100 # 100
    innerdims = dims
    negrate = 10
    numbats = 100 # 100
    epochs = 100 #20
    wreg = 0.0000001
    lr = 0.01/numbats #0.0001 # for SGD
    lr2 = 1.
    evalinter = 1
    rho = 0.95


    ############"
    dims = 20
    innerdims = dims#50
    lr = 0.001/numbats # 8

    toy = False

    threshold = 0.5
    #paths
    datatt = TT("data")

    if toy:
        dims = 10
        numbats=10
        wreg = 0.0
        lr=0.1/numbats
        datafileprefix = "../../data/"
        tensorfile = "toy.ssd"
        vocabsize=11
        epochs=100
        numrels = 1
    else:
        # get the data and split
        datafileprefix = "../../data/nycfilms/triples.flat/"
        tensorfile = "alltripletensor.flat.ssd"
        fulldic = loaddic(datafileprefix+"tripletensor.flatidx.pkl")
        vocabsize = len(fulldic)
        numrels = 20

    innerdim2 = 20

    data = loaddata(datafileprefix+tensorfile)
    data = data.keys.lok
    trainX = data[:, :2]
    labels = data[:, -1]
    # labels = data[:, 1:]
    datatt.tock("loaded")

    # train model
    model = AddEKMM(numrels=numrels, dim=dims, vocabsize=vocabsize, maxiter=epochs, wreg=wreg, numbats=numbats, negrate=negrate, validsplit=0.02)\
                .autosave.normalize \
            + SGD(lr=lr) \
            #+ IFGRU(dim=dims, innerdim=innerdims, wreg=wreg)
    err, verr = model.train(trainX, labels, evalinter=evalinter)
    erfile = "allcompat.flat.ssd"
    #traincompat(model.W.get_value(), erfile)
    plt.plot(err, "r")
    if len(verr) > 0:
        plt.plot(verr, "g")
    plt.show(block=True)


    #model.save(getsavepath())
    '''print "test prediction:"
    print model.getpredictfunction()(11329, 9325, 7156)
    print model.getpredictfunction()(11329, 3674, 7155)'''
    if toy:
        print model.predict(0, 10, 1)
        print model.predict(0, 10, 2)
    else:
        print model.predict([417], [[11307]], [9145])
        print model.predict([417], [[11307]], [9156])

    #embed()

def dotraincompat():
    emodel = EKMM.load("../../models/MatMulEKMM+SGD+n10+E20D.2016-01-06=16:21.auto")
    entemb = emodel.W.get_value()
    erfile = "allcompat.flat.ssd"
    traincompat(entemb, erfile)

def traincompat(entemb, erfile): # entemb: (vocabsize, dim) matrix of entity embeddings
                                 # erfile: path to file containing which entity has which relation
    # params
    negrate = 3
    numbats = 100 # 100
    epochs = 200 #20
    wreg = 0.0000001
    evalinter = 1
    lr = 0.001/numbats # 8

    toy = False

    tt = TT("data")

    if toy:
        dims = 10
        numbats=10
        wreg = 0.0
        lr=0.1/numbats
        datafileprefix = "../../data/"
        tensorfile = "toy.ssd"
        vocabsize=11
        epochs=100
    else:
        # get the data and split
        datafileprefix = "../../data/nycfilms/triples.flat/"
        fulldic = loaddic(datafileprefix+"compatreldic.flatidx.pkl")
        vocabsize = len(fulldic)

    data = loaddata(datafileprefix+erfile)
    data = data.keys.lok
    trainX = data[:, :1]
    labels = data[:, 1]
    tt.tock("loaded")

    # train model
    model = FixedEntCompat(entembs=entemb, vocabsize=vocabsize, maxiter=epochs, wreg=wreg, numbats=numbats, negrate=negrate)\
                .autosave.normalize \
            + SGD(lr=lr)
    err = model.train(trainX, labels, evalinter=evalinter)
    plt.plot(err, "r")
    plt.show(block=True)


class FixedEntCompat(EKMM): # margin-based matrix factorization
    def __init__(self, entembs, **kw):
        self.entembs = theano.shared(entembs)
        kw.update([("dim", entembs.shape[1])])
        super(FixedEntCompat, self).__init__(**kw)

    def defmodel(self):
        lhs = T.ivector("lhs")
        rhs, nrhs = T.ivectors("rhs","nrhs")
        lhsemb = self.entembs[lhs, :]
        rhsemb = self.W[rhs, :]
        nrhsemb = self.W[nrhs, :]
        pdot = T.batched_dot(lhsemb, rhsemb)
        ndot = T.batched_dot(lhsemb, nrhsemb)
        return pdot, ndot, [lhs, rhs, nrhs]

    def getsamplegen(self, trainX, labels):
        innersamplegen = super(FixedEntCompat, self).getsamplegen(trainX, labels)
        def samplegen():
            innerret = innersamplegen()
            return [innerret[0], innerret[2], innerret[3]]
        return samplegen

    def getpredictfunction(self):
        pdot, _, inps = self.defmodel()
        scoref = theano.function(inputs=[inps[0], inps[2]], outputs=pdot)
        def pref(l, r):
            args = [np.asarray(i).astype("int32") for i in [l, r]]
            return scoref(*args)
        return pref


###################### FUNCTIONS FOR INSPECTION ##########################

def extend(instance, new_class):
    instance.__class__ = type(
                '%s_extended_with_%s' % (instance.__class__.__name__, new_class.__name__),
                (instance.__class__, new_class),
                {}
            )


def loadmodel(path):
    return RNNTFSGDSM.load(path)


def loaddicts(path):
    dicts = pickle.load(open(path))
    xdic = dicts["xdic"]
    ydic = dicts["ydic"]
    zdic = dicts["zdic"]
    return xdic, ydic, zdic


def transform(vect, mat):
    return np.dot(vect, mat)


def getcompat(vectA, vectB):
    return np.dot(vectA, vectB)


def getcompati(idxA, idxB, m):
    vectA = m.embedX(idxA)
    vectB = m.embedY(idxB)
    return getcompat(vectA, vectB)


def chaintransform(model, idx, *tidxs):
    vect = model.embedX(idx)
    for id in tidxs:
        vect = transform(vect, model.embedZ(id))
    return vect


if __name__ == "__main__":
    run()
    #dotraincompat()