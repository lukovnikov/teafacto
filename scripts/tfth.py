__author__ = 'denis'

from teafacto.data.sptensor import SparseTensor
from datetime import datetime
from matplotlib import pyplot as plt
import pickle

from teafacto.tensorfac import TFSGDC

# TODO:  load all data in a sparse tensor
#        - reverse trans ==> reverse triple
#        - map to the same id space
#        - reshape data


def loaddata(file):
    st = SparseTensor.from_ssd(file)
    return st

def run():
    # params
    dims = 15
    negrate = 1
    numbats = 1000
    epochs = 50 #20
    wreg = 0.001
    lr = 0.0001 #0.0001
    evalinter = 1

    #paths
    datafileprefix = "../data/nycfilms/"
    dirfwdsuffix = "trans_rev_rev.ssd"

    # get the data and split
    start = datetime.now()
    data = loaddata(datafileprefix+dirfwdsuffix)
    print "source data loaded in %f seconds" % (datetime.now() - start).total_seconds()
    start = datetime.now()

    numslices, numrows, numcols = data.shape

    # train model
    print "training model"
    start = datetime.now()
    model = TFSGDC(dims=dims, maxiter=epochs, lr=lr, wregs=wreg, numbats=numbats)
    print "model defined in %f" % (datetime.now() - start).total_seconds()
    start = datetime.now()
    W, R, err = model.train(data, evalinter=evalinter)
    print "model trained in %f" % (datetime.now() - start).total_seconds()
    print len(err)
    plt.plot(err, "r")
    plt.show(block=False)

if __name__ == "__main__":
    run()