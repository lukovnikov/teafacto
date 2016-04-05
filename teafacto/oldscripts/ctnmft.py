__author__ = 'denis'

from datetime import datetime
from collections import OrderedDict, Counter
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython import embed

from teafacto.facs.mf import NMFSGDCL


#from teafacto.mftf import NMFALS, NMFSGDC


np.random.seed(12345)


def readdata(inf):
    data = [] # list of tuples
    # read file line by line
    for _, row in inf.iterrows():
        for rhse in row.r:
            data.append((row.e, rhse))
    datadf = pd.DataFrame(data)
    return datadf


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
    dirfwdsuffix = "direct_both_typ.pd.pkl"

    # get the data and split
    start = datetime.now()
    srcdf = pickle.load(open(datafileprefix+dirfwdsuffix))
    datadf = readdata(srcdf)
    print "source data loaded in %f" % (datetime.now() - start).total_seconds()
    start = datetime.now()

    numents = int(datadf.ix[:, 0].max())+1
    #print numents
    numrels = int(datadf.ix[:, 1].max())+1
    #print numrels

    datamatdf = pd.DataFrame.from_records(srcdf["r"].apply(Counter)).fillna(value=0)
    datamatdf["e"] = srcdf["e"]
    datamatdf.set_index("e", inplace=True)

    # build intermediate dics
    imentdic = dict(zip(range(len(datamatdf.index.values)), datamatdf.index.values))
    imreldic = dict(zip(range(len(datamatdf.columns)), datamatdf.columns))
    revimentdic = {v: k for k, v in imentdic.items()}
    revimreldic = {v: k for k, v in imreldic.items()}
    print "data transformed"

    # train model
    print "training model"
    start = datetime.now()
    #model = NMFSGDC(dims=dims, maxiter=epochs, lr=lr, Lreg=wreg, Rreg=wreg, numbats=numbats)
    model = NMFSGDCL(dims=dims, maxiter=epochs, lr=lr, Lreg=wreg, Rreg=wreg*5, numbats=numbats)
    #model = NMFALS(dims=dims, maxiter=100)
    #model = NMFGD(dims=dims, maxiter=epochs, lr=lr, Lreg=wreg, Rreg=wreg)
    print "model defined in %f" % (datetime.now() - start).total_seconds()
    start = datetime.now()
    W, H, err = model.train(datamatdf.values.astype("float32"), evalinter=evalinter)
    #W, H, err = model.train(datadf.values.astype("float32"), numents, numrels, evalinter=evalinter)
    print "model trained in %f" % (datetime.now() - start).total_seconds()
    print len(err)
    plt.plot(err, "r")
    plt.show(block=False)


    #nexplore(model.predict, lambda x: revimentdic[x], lambda x: revimreldic[x])

    explore(W, H.T, lambda x: revimentdic[x], lambda x: revimreldic[x])

    embed()

def nexplore(pfun, edicf=lambda x:x, rdicf=lambda x:x):
    ents = [9105] * 6
    rels = [31,532,27,528,409,910]
    print pfun(zip(map(edicf, ents), map(rdicf, rels)))

def explore(emb, rmbf, edicf=lambda x: x, rdicf=lambda x: x):
    all = OrderedDict()
    all["ultron"] = emb[edicf(9105)]
    all["window"] = emb[edicf(4811)]
    all["stone"] = emb[edicf(6131)]
    all["alfie"] = emb[edicf(94)]
    all["grace"] = emb[edicf(3031)]
    all["warner"] = emb[edicf(2380)]
    all["direc"] = rmbf[rdicf(31)]
    all["-direc"] = rmbf[rdicf(532)]
    all["spouse"] = rmbf[rdicf(27)]
    all["-spouse"] = rmbf[rdicf(528)]
    all["found"] = rmbf[rdicf(409)]
    all["-found"] = rmbf[rdicf(910)]
    datadf = buildcrossmat(all, f = cosine)
    print datadf.to_string(float_format=lambda x: "%.2f" % x)


def buildcrossmat(dic, f):
    data = [[f(dic[x], dic[y]) for y in dic] for x in dic]
    datadf = pd.DataFrame(data)
    datadf.columns = dic.keys()
    datadf["idx"] = dic.keys()
    datadf["anorms"] = [np.linalg.norm(dic[x]) for x in dic]
    datadf.set_index("idx", inplace=True)
    return datadf


def dot(x, y):
    return np.dot(x, y)


def cosine(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))



def getonrun():
    def onrun(state):
        state.plotpos = []
        state.plotneg = []
        state.plotloss = []
        state.plotvloss = []
        #state.lossum = []
        state.plotr2rpos = []
        state.plotr2rneg = []
    return onrun


def getoffepoch(outtest, rel2rel):
    def offepoch(state):
        state.plotloss.append(state._epochloss)
        state.plotvloss.append(state._validloss)
        #state.lossum.append(state._validloss + state._epochloss)
        pos = outtest(np.asarray([[9105, 31]], dtype="int32"))[0] # Ultron's :director
        state.plotpos.append(pos)
        neg = outtest(np.asarray([[9105, 409]], dtype="int32"))[0] # Ultron's :foundedBy
        r2rpos = rel2rel(np.asarray([[535, 31]], dtype="int32"))
        r2rneg = rel2rel(np.asarray([[535, 409]], dtype="int32"))
        state.plotr2rpos.append(r2rpos)
        state.plotr2rneg.append(r2rneg)
        state.plotneg.append(neg)
        pstr = "Epoch %d \t - Loss: %f" % (state._epoch, state._epochloss)
        if state._validloss is not None:
            pstr += " \t - %f" % (state._validloss)
        print pstr
    return offepoch


def offrun(state):
    plt.figure(1)
    plt.subplot(221)
    vlossx = []
    vlossy = []
    vlxc = 0
    for x in state.plotvloss:
        if x is not None:
            vlossx.append(vlxc)
            vlossy.append(x)
        vlxc += 1
    plt.plot(range(len(state.plotloss)), state.plotloss, "m",
             vlossx, vlossy, "c")
             # range(len(state.lossum)), state.lossum, "k")
    plt.subplot(222)
    plt.plot(range(len(state.plotpos)), state.plotpos, "g",
             range(len(state.plotneg)), state.plotneg, "r")
    plt.subplot(223)
    plt.plot(range(len(state.plotr2rpos)), state.plotr2rpos, "g",
             range(len(state.plotr2rneg)), state.plotr2rneg, "r")
    plt.show(block=False)


def transbat(batch):
    return (batch[:, :].astype("int32"), np.ones((batch.shape[0],), dtype="float32"))


def getonbatch(negrate, srcdf, numrels):
    def onbatch(state):
        ret = state.batchsamples
        ret = tuple(map(lambda x: x.astype("int32"), ret))
        neg = None
        for i in range(0,negrate):
            addneg = np.array(np.random.randint(0, numrels, (ret[0].shape[0], 1)), dtype="int32")
            if neg is None:
                neg = addneg
            else:
                neg = np.concatenate((neg, addneg), axis=0)
        zeroes = -1. * np.ones((ret[0].shape[0]*negrate,), dtype="float32")
        negrep = np.repeat(ret[0][:, 0:1], negrate, axis=0)
        negs = np.concatenate((negrep, neg), axis=1)
        state.batchsamples = (np.concatenate((ret[0], negs), axis=0),
                              np.concatenate((ret[1], zeroes), axis=0).astype("float32"))
    return onbatch


if __name__ == "__main__":
    run()