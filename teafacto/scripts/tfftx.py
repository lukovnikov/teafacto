__author__ = 'denis'

from teafacto.data.sptensor import SparseTensor
from datetime import datetime
from matplotlib import pyplot as plt
import pickle, os, pkgutil, pandas as pd, numpy as np
from IPython import embed

from teafacto.tf import TFSGDC, TFMF0SGDC, TFMXSGDC, RESCALSGDC


def loaddata(file):
    file = os.path.join(os.path.dirname(__file__), file)
    st = SparseTensor.from_ssd(file)
    return st

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
    dims = 20
    negrate = 1
    numbats = 100
    epochs = 1000 #20
    wreg = [0.007, 0.001, 0.000000001]
    lr = .03/numbats #0.0001
    lr2 = 0.00005
    evalinter = 1

    threshold = 0.5
    #paths
    datafileprefix = "../../data/nycfilms/"
    #tensorfile = "fulltensorext.ssd"
    tensorfile = "tripletensor.ssd"

    # get the data and split
    start = datetime.now()
    data = loaddata(datafileprefix+tensorfile)
    data.threshold(threshold)

    print "source data loaded in %f seconds" % (datetime.now() - start).total_seconds()

    numslices, numrows, numcols = data.shape

    # train model
    print "training model"
    start = datetime.now()
    #model = TFSGDC(dims=dims, maxiter=epochs, lr=lr, wregs=wreg, numbats=numbats, corruption="rhs").autosave
    #model = TFMF0SGDC(dims=dims, maxiter=epochs, lr=lr, wregs=wreg, numbats=numbats, corruption="rhs", lr2=lr2, invZoffset=501).autosave
    model = TFMXSGDC(dims=dims, maxiter=epochs, lr=lr, wregs=wreg, numbats=numbats, corruption="nmhs", invZoffset=501).autosave
    #model = RESCALSGDC(dims=dims, maxiter=epochs, lr=lr, wregs=wreg, numbats=numbats, corruption="nmhs").autosave
    print "model %s defined in %f" % (model.__class__.__name__, (datetime.now() - start).total_seconds())
    start = datetime.now()
    params, err = model.train(data, evalinter=evalinter)
    print "model trained in %f" % (datetime.now() - start).total_seconds()
    print len(err)
    plt.plot(err, "r")
    plt.show(block=False)

    model.save(getsavepath())

    embed()


###################### FUNCTIONS FOR INSPECTION ##########################

def extend(instance, new_class):
    instance.__class__ = type(
                '%s_extended_with_%s' % (instance.__class__.__name__, new_class.__name__),
                (instance.__class__, new_class),
                {}
            )


def loadmodel(path):
    model = TFSGDC.load(path)
    return model


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