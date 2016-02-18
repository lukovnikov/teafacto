from teafacto.index.wikiindex import WikipediaIndex
from teafacto.qa.multichoice.kaggleutils import *
from teafacto.core.utils import ticktock, argparsify
import multiprocessing as mp, pandas as pd, math, numpy as np
from teafacto.lm import Glove
from IPython import embed
import itertools

def choose(sdf): # sdf only has qA, qB, qC, qD and the right index (question id's)
    ret = pd.DataFrame()
    ret["correctAnswer"] = sdf.idxmax(axis=1).apply(lambda x: x[1])
    ret.index = sdf.index
    return ret

def evalu(pred, orig):
    return (pred["correctAnswer"] == orig["correctAnswer"]).sum(axis=0)*1. / orig.shape[0]

def getencoder(g=None, wgetter=None):
    if g is None:
        g = Glove(50)
    def encode(l):
        el = np.asarray([(g % x)*wgetter(x) for x in l])
        el = el.sum(axis=0)
        el = el/(np.linalg.norm(el) + 10e-15)
        return el
    return encode

def getavgdotenc(g=None, wgetter=None):
    if g is None:
        g = Glove(50)
    def encode(l):
        el = np.asarray([(g % x)*wgetter(x) for x in l])
        el = el.sum(axis=0)
        el = el/(np.linalg.norm(el) + 10e-15)
        return el
    return encode

def getdotsumscore(g=None, wgetter=None):
    if g is None:
        g = Glove(50)
    def dotsumscore(row):
        def encode(l):
            el = np.asarray([(g % x) * wgetter(x) for x in l])
            el = el.sum(axis=0)
            el = el/(np.linalg.norm(el) + 10e-15)
            return el
        def score(x, y):
            return np.dot(x, y)
        row["question"] = encode(row["question"])
        row["answerA"] = encode(row["answerA"])
        row["answerB"] = encode(row["answerB"])
        row["answerC"] = encode(row["answerC"])
        row["answerD"] = encode(row["answerD"])
        return pd.Series(data=[ score(row["question"], row["answerA"]),
                                score(row["question"], row["answerB"]),
                                score(row["question"], row["answerC"]),
                                score(row["question"], row["answerD"]) ],
                         index=["qA", "qB", "qC", "qD"])
    return dotsumscore

def getsumdotscore(g=None, wgetter=None):
    if g is None:
        g = Glove(50)
    def sumdotscore(row):
        ql = row["question"]
        qa = [x for x in itertools.product(ql, row["answerA"])]
        qb = [x for x in itertools.product(ql, row["answerB"])]
        qc = [x for x in itertools.product(ql, row["answerC"])]
        qd = [x for x in itertools.product(ql, row["answerD"])]
        #embed()
        qa, qb, qc, qd = map(lambda x:
                             reduce(lambda k, l: k+l,
                                    map(lambda y: wgetter(y[0])*wgetter(y[1])*np.dot(g % y[0], g % y[1]), x),
                                    0) / (len(x)+10e-15),
                             [qa, qb, qc, qd])
        return pd.Series(data=[qa, qb, qc, qd], index=["qA", "qB", "qC", "qD"])
    return sumdotscore

def getmaxdotscore(g=None, wgetter=None):
    if g is None:
        g = Glove(50)
    def sumdotscore(row):
        ql = row["question"]
        qa = [x for x in itertools.product(ql, row["answerA"])]
        qb = [x for x in itertools.product(ql, row["answerB"])]
        qc = [x for x in itertools.product(ql, row["answerC"])]
        qd = [x for x in itertools.product(ql, row["answerD"])]
        #embed()
        qa, qb, qc, qd = map(lambda x:
                             reduce(lambda k, l: max(k, l),
                                    map(lambda y: wgetter(y[0])*wgetter(y[1])*np.dot(g % y[0], g % y[1]), x),
                                    0) / (len(x)+10e-15),
                             [qa, qb, qc, qd])
        return pd.Series(data=[qa, qb, qc, qd], index=["qA", "qB", "qC", "qD"])
    return sumdotscore

def getdefwgetter():
    def wgetter(a):
        return 1.
    return wgetter

def getidxwgetter(idx):
    def wgetter(a):
        return idx.getidf(a)
    return wgetter

def run(dims=50, path=None, widx=False):
    g = Glove(dims)
    if path is None:
        d = read()
    else:
        d = read(path)

    if widx is False:
        wg = getdefwgetter()
    elif widx is True:
        irp = WikipediaIndex()
        wg = getidxwgetter(irp)
    else:
        irp = WikipediaIndex(dir=widx)
        wg = getidxwgetter(irp)

    sdf = d.apply(getsumdotscore(g=g, wgetter=wg), axis=1)

    '''
    encode = getencoder(g, wgetter=wg)
    d["question"] = d["question"].map(encode)
    d["answerA"] = d["answerA"].map(encode)
    d["answerB"] = d["answerB"].map(encode)
    d["answerC"] = d["answerC"].map(encode)
    d["answerD"] = d["answerD"].map(encode)

    qa = d.apply(lambda x: score(x["question"], x["answerA"]), axis=1)
    qb = d.apply(lambda x: score(x["question"], x["answerB"]), axis=1)
    qc = d.apply(lambda x: score(x["question"], x["answerC"]), axis=1)
    qd = d.apply(lambda x: score(x["question"], x["answerD"]), axis=1)

    sdf = pd.DataFrame({"qA":qa, "qB":qb, "qC":qc, "qD":qd})
    '''
    print evalu(choose(sdf), d)


if __name__ == "__main__":
    run(**argparsify(run))