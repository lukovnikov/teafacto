__author__ = 'denis'

from teafacto.data.sptensor import SparseTensor
from datetime import datetime
from matplotlib import pyplot as plt
import pickle, os, pkgutil, pandas as pd, numpy as np
from IPython import embed

from teafacto.rnnkm import RNNTFSGDSM
from teafacto.xrnnkm import RNNEKMM, RNNEOKMM, AddEKMM, DiagMulEKMM, RNNEKMSM, AutoRNNEKMSM, RNNESMSM, RNNESMSMShort
from teafacto.optimizers import SGD, RMSProp, AdaDelta
from teafacto.rnn import GRU, LSTM, RNU


def loaddata(file):
    file = os.path.join(os.path.dirname(__file__), file)
    data = pickle.load(open(file))
    return data

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
    epochs = np.infty #20
    wreg = 0.0000001
    lr = 0.01/numbats #0.0001 # for SGD
    lr2 = 1.
    evalinter = 1
    rho = 0.95
    occ = 0.3


    ############"
    dims = 50
    innerdims = 400
    lr = 1./numbats # 8

    start = datetime.now()

    # get the data and split
    datafileprefix = "../../data/quotes/"
    tensorfile = "quotes.np.pkl"
    vocabsize = 128 # ascii characters

    data = loaddata(datafileprefix+tensorfile)

    trainX = data[:, :-1]
    labels = data[:, 1:]

    print "source data loaded in %f seconds" % (datetime.now() - start).total_seconds()

    # train model
    print "training model"
    start = datetime.now()
    model = RNNESMSM(dim=dims, vocabsize=vocabsize, maxiter=epochs, wreg=wreg, numbats=numbats, negrate=negrate)\
                .autosave.normalize \
            + SGD(lr=lr) \
            + GRU(dim=dims, innerdim=innerdims, wreg=wreg)
    print "model %s defined in %f" % (model.__class__.__name__, (datetime.now() - start).total_seconds())
    start = datetime.now()
    err = model.train(trainX, labels, evalinter=evalinter)
    print "model trained in %f" % (datetime.now() - start).total_seconds()
    plt.plot(err, "r")
    plt.show(block=True)

    print model.predict([[34, 65]], [67])
    print model.predict([[34, 65]], [34])

    #embed()


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