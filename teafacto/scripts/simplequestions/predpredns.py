from teafacto.scripts.simplequestions.fullrank import readdata, SeqEncDecRankSearch, FullRankEval, shiftdata, EntEncRep, EntEnc, EntEmbEnc
from teafacto.util import argprun, ticktock
import numpy as np, os, sys, math
from IPython import embed
from teafacto.core.base import Val, tensorops as T, Block

from teafacto.blocks.seq.enc import SimpleSeq2Vec, SimpleSeq2Sca, SeqUnroll
from teafacto.blocks.match import SeqMatchScore, MatchScore, GenDotDistance, DotDistance, CosineDistance
from teafacto.blocks.basic import VectorEmbed, MatDot


def run(
        epochs=50,
        mode="char",  # "char" or "word" or "charword"
        numbats=1000,
        lr=0.1,
        wreg=0.000001,
        bidir=False,
        layers=1,
        encdim=200,
        decdim=200,
        embdim=100,
        negrate=1,
        margin=1.,
        hingeloss=False,
        debug=False,
        preeval=False,
        sumhingeloss=False,
        checkdata=False,  # starts interactive shell for data inspection
        printpreds=False,
        subjpred=False,
        predpred=False,
        specemb=-1,
        usetypes=False,
        evalsplits=50,
        cosine=False,
        loadmodel=False,
):
    if debug:  # debug settings
        hingeloss = True
        numbats = 10
        lr = 0.02
        epochs = 1
        printpreds = True
        whatpred = "all"
        if whatpred == "pred":
            predpred = True
        elif whatpred == "subj":
            subjpred = True
        preeval = True
        # specemb = 100
        margin = 1.
        evalsplits = 1
        # usetypes=True
        # mode = "charword"
        # checkdata = True


    # load the right file
    maskid = -1
    tt = ticktock("script")
    specids = specemb > 0
    tt.tick()
    (traindata, traingold), (validdata, validgold), (testdata, testgold), \
    worddic, entdic, entmat, relstarts, canids, wordmat, chardic \
        = readdata(mode, testcans="testcans.pkl", debug=debug, specids=True,
                   usetypes=usetypes, maskid=maskid)
    entmat = entmat.astype("int32")
    # transform for predpred
    traingold = traingold[:, 1] - relstarts
    validgold = validgold[:, 1] - relstarts
    testgold = testgold[:, 1] - relstarts

    if checkdata:
        rwd = {v: k for k, v in worddic.items()}
        red = {v: k for k, v in entdic.items()}

        def p(xids):
            return (" " if mode == "word" else "").join([rwd[xid] if xid > -1 else "" for xid in xids])

        embed()

    print traindata.shape, traingold.shape, testdata.shape, testgold.shape

    tt.tock("data loaded")

    numwords = max(worddic.values()) + 1
    numents = max(entdic.values()) + 1
    print "%d words, %d entities" % (numwords, numents)

    if bidir:
        encinnerdim = [encdim / 2] * layers
    else:
        encinnerdim = [encdim] * layers

    memembdim = embdim
    memlayers = layers
    membidir = bidir
    if membidir:
        decinnerdim = [decdim / 2] * memlayers
    else:
        decinnerdim = [decdim] * memlayers

    emb = VectorEmbed(numwords, embdim)
    predemb = VectorEmbed(numents - relstarts + 1, decinnerdim[-1])
    inpenc = SimpleSeq2Vec(inpemb=emb,
                           inpembdim=emb.outdim,
                           innerdim=encinnerdim,
                           maskid=maskid,
                           bidir=bidir,
                           layers=layers)

    dist = DotDistance() if not cosine else CosineDistance()
    scorerkwargs = {"argproc": lambda x, y: ((x,), (y,)),
                    "scorer": dist}
    scorer = MatchScore(inpenc, predemb, **scorerkwargs)


    class NegIdxGen(object):
        def __init__(self, rng):
            self.min = 0
            self.max = rng

        def __call__(self, datas, gold):
            predrand = np.random.randint(self.min, self.max, (gold.shape[0],))
            return datas, predrand.astype("int32")


    # embed()

    obj = lambda p, n: n - p
    if hingeloss:
        obj = lambda p, n: (n - p + margin).clip(0, np.infty)

    tt.tick("training")
    nscorer = scorer.nstrain([traindata, traingold]) \
        .negsamplegen(NegIdxGen(numents - relstarts))\
        .negrate(negrate).objective(obj) \
        .adagrad(lr=lr).l2(wreg).grad_total_norm(1.0) \
        .validate_on([validdata, validgold]) \
        .train(numbats=numbats, epochs=epochs)
    tt.tock("trained")

    # eval
    canids = np.arange(start=0, stop=numents-relstarts)
    predembs = predemb.predict(canids)   # (numrels, embdim)
    tt.tick("evaluating")
    predencs = inpenc.predict(testdata)     # (batsize, embdim)
    scores = np.zeros((predencs.shape[0], predembs.shape[0]))
    for i in range(predencs.shape[0]):
        scores[i, :] = scorer.s.predict(np.repeat(predencs[np.newaxis, i], predembs.shape[0], axis=0),
                                        predembs)[0]
        tt.progress(i, predencs.shape[0], live=True)
    best = np.argmax(scores, axis=1)
    accuracy = np.sum(best == testgold)*1. / testgold.shape[0]
    print accuracy
    embed()

    tt.tock("evaluated")


if __name__ == "__main__":
    argprun(run)