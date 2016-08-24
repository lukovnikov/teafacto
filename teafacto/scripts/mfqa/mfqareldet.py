from collections import OrderedDict

import numpy as np

from teafacto.blocks.seq.oldseqproc import SimpleSeq2Idx
from teafacto.util import argprun


def readdic(dicp):
    dic = OrderedDict()
    with open(dicp) as f:
        for line in f:
            k, v = line.split("\t")
            v = int(v)
            dic[k] = v
    return dic

def readdatamat(p, top=np.infty):
    data = []
    gold = []
    c = 0
    rows = None
    cols = None
    with open(p) as f:
        for line in f:
            if rows is None and cols is None:
                rows, cols = map(int, line.split())
                data = np.zeros((min(rows, top), cols), dtype="int32") - 1
                gold = np.zeros((min(rows, top), ), dtype="int32")
                continue
            s, r = line.split("\t")
            r = int(r)
            s = map(int, s.split())
            data[c, :len(s)] = s
            gold[c] = r
            if top is not None and c >= top-1:
                break
            c += 1
    return data, gold


def datasplit(target, *xs, **kw):
    random = kw["random"] if "random" in kw else False
    splits = kw["splits"] if "splits" in kw else 5
    xs = (target,) + xs
    assert([x.shape[0] for x in xs].count(xs[0].shape[0]) == len(xs))
    batsize = xs[0].shape[0]
    splitidxs = range(batsize)
    if random:
        np.random.shuffle(splitidxs)
    idxsplit = batsize // splits
    holdoutidxs = splitidxs[-idxsplit:]
    keepinidxs = splitidxs[:idxsplit*(splits-1)]
    holdout = [x[holdoutidxs] for x in xs]
    keepin = [x[keepinidxs] for x in xs]
    return tuple(keepin), tuple(holdout)


def smartdatasplit(target, *xs, **kw):
    """ constructs minimal holdout dataset with all classes. discards classes with too few examples"""
    random = kw["random"] if "random" in kw else False
    keepmincount = kw["keepmincount"] if "keepmincount" in kw else 1
    holdmincount = kw["holdmincount"] if "holdmincount" in kw else 1
    xs = (target,) + xs
    assert([x.shape[0] for x in xs].count(xs[0].shape[0]) == len(xs))
    batsize = xs[0].shape[0]
    globcounts = {}
    # gather class usage stats
    for i in range(batsize):
        k = target[i]
        if k not in globcounts:
            globcounts[k] = 0
        globcounts[k] += 1
    # create new datas
    keepsize = 0
    holdsize = 0
    holdcounts = {}
    keepcounts = {}
    for k in globcounts:
        if globcounts[k] >= keepmincount + holdmincount:
            holdsize += holdmincount
            holdcounts[k] = holdmincount
            keepsize += globcounts[k] - holdmincount
            keepcounts[k] = globcounts[k] - holdmincount
    keepxs = [np.zeros((keepsize,) + x.shape[1:], dtype=x.dtype) for x in xs]
    holdxs = [np.zeros((holdsize,) + x.shape[1:], dtype=x.dtype) for x in xs]
    # populate datas
    idxs = np.arange(0, batsize)
    if random:
        np.random.shuffle(idxs)
    kidx = 0
    hidx = 0
    for i in range(batsize):
        idx = idxs[i]
        tidx = target[idx]
        if tidx in holdcounts:
            if holdcounts[tidx] > 0:
                holdcounts[tidx] -= 1
                for x, y in zip(holdxs, xs):
                    x[kidx, ...] = y[idx, ...]
                kidx += 1
            elif keepcounts[tidx] > 0:
                keepcounts[tidx] -= 1
                for x, y in zip(keepxs, xs):
                    x[hidx, ...] = y[idx, ...]
                hidx += 1
            else:
                print "sum ting wong"
    return tuple(keepxs), tuple(holdxs)


def evaluate(pred, gold):
    return np.sum(gold == pred) * 100. / gold.shape[0]




def run(
        epochs=10,
        numbats=100,
        numsam=10000,
        lr=0.1,
        wdicp="../../../data/mfqa/clean/reldet/mfqa.reldet.worddic.sample.big",
        rdicp="../../../data/mfqa/clean/reldet/mfqa.reldet.reldic.sample.big",
        datap="../../../data/mfqa/clean/reldet/mfqa.reldet.sample.big", # 65% - 63%, 64.2% - 65.37%, 67% - 66.3%
        embdim=100,
        innerdim=200,
        wreg=0.00005,
        bidir=False,
        keepmincount=5,
        ):
    #wdic = readdic(wdicp)
    #rdic = readdic(rdicp)

    data, gold = readdatamat(datap, numsam)
    print data.shape, gold.shape

    (traingold, traindata), (testgold, testdata) = \
        smartdatasplit(gold, data, random=True, keepmincount=keepmincount, holdmincount=2)

    print traindata.shape, testdata.shape

    numwords = np.max(data)+1
    numrels = np.max(gold)+1
    print numwords, numrels

    if bidir:
        innerdim /= 2

    m = SimpleSeq2Idx(
        indim=numwords,
        outdim=numrels,
        inpembdim=embdim,
        innerdim=innerdim,
        maskid=-1,
        bidir=bidir,
    )

    m = m.train([traindata], traingold).adagrad(lr=lr).l2(wreg).grad_total_norm(1.0).cross_entropy()\
        .split_validate(10, random=True).accuracy().cross_entropy().takebest()\
        .train(numbats=numbats, epochs=epochs)

    pred = m.predict(testdata)
    print pred.shape
    evalres = evaluate(np.argmax(pred, axis=1), testgold)
    print str(evalres) + "%"


if __name__ == "__main__":
    argprun(run)