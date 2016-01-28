__author__ = 'denis'

from teafacto.data.sptensor import SparseTensor
from datetime import datetime
from matplotlib import pyplot as plt
import pickle, os, pkgutil, pandas as pd, numpy as np

from teafacto.tensorfac import TFSGDC

# TODO:  - 1. do the training
#           - first check if sampling works correctly
#           - then check if training works correctly


def loaddata(file):
    file = os.path.join(os.path.dirname(__file__), file)
    st = SparseTensor.from_ssd(file)
    return st

def loadmeta(dfile):
    dfile = os.path.join(os.path.dirname(__file__), dfile)
    meta = pickle.load(open(dfile))
    return meta

def run():
    # params
    dims = 15
    negrate = 1
    numbats = 1000
    epochs = 50 #20
    wreg = 0.001
    lr = 0.0001 #0.0001
    evalinter = 1

    threshold = 0.5
    #paths
    datafileprefix = "../data/nycfilms/"
    tensorfile = "fulltensor.ssd"
    metafile = "fulltensor.apx.pkl"

    # get the data and split
    start = datetime.now()
    data = loaddata(datafileprefix+tensorfile)
    data.threshold(threshold)

    meta = loadmeta(datafileprefix+metafile)
    fulldic = meta["xydic"]
    reldic = meta["zdic"]
    splitid = meta["xyrelsplit"]
    print "source data loaded in %f seconds" % (datetime.now() - start).total_seconds()
    start = datetime.now()

    numslices, numrows, numcols = data.shape

    # train model
    print "training model"
    start = datetime.now()
    model = TFSGDC(dims=dims, maxiter=epochs, lr=lr, wregs=wreg, numbats=numbats, wsplit=splitid)
    print "model defined in %f" % (datetime.now() - start).total_seconds()
    start = datetime.now()
    W, R, err = model.train(data, evalinter=evalinter)
    print "model trained in %f" % (datetime.now() - start).total_seconds()
    print len(err)
    plt.plot(err, "r")
    plt.show(block=False)

    inspect(model, fulldic, reldic)

def inspect(model, fulldic, reldic):
    dbr = "http://dbpedia.org/resource/"
    dbo = "http://dbpedia.org/ontology/"
    dbp = "http://dbpedia.org/property/"
    lhs = [
        "Emma_Stone",
        "Alfred_Hitchcock",
        "Grace_Kelly",
        "Warner_Bros.",
        "Rear_Window",
        "Avengers:_Age_of_Ultron"
    ]
    relt = [
        "starring",
        "director",
        "spouse",
    ]
    rhs = [
        "starring",
        "starring-",
        "producer",
        "producer-",
        "director",
        "director-",
        "spouse",
        "spouse-"
    ]
    lhsx = map(lambda x: dbr + x, lhs)
    reltx = map(lambda x: dbo + x, relt)
    rhsx = map(lambda x: "-" + dbo + x[:-1] if x[-1] == "-" else dbo + x, rhs)
    lhsx = map(lambda x: fulldic[x], lhsx)
    reltcidx = map(lambda x: fulldic[x], reltx)
    reltx = map(lambda x: reldic[x], reltx)
    rhsx = map(lambda x: fulldic[x], rhsx)
    pfun = model.getpredf()
    for i, relte in enumerate(reltx):
        print "\n-------------------------------------"
        print relt[i]
        inspmat = buildinspmat(relte, lhsx, rhsx, lambda x, y, z: pfun(x, z, y))
        inspmat["idx"] = lhs
        inspmat.set_index("idx", inplace=True)
        inspmat.columns = rhs
        print inspmat.to_string(float_format=lambda x: "%.2f" % x)

def buildinspmat(relte, lhs, rhs, f):
    data = [f(
        np.asarray([x]*len(rhs)).astype("int32"),
        np.asarray(rhs).astype("int32"),
        np.asarray([relte]*len(rhs)).astype("int32")
    )[0] for x in lhs]
    datadf = pd.DataFrame(data)
    return datadf


if __name__ == "__main__":
    run()
