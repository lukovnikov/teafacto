from teafacto.blocks.seq.enc import SimpleSeq2Bool, SimpleSeq2Vec, SimpleSeq2Idx
from teafacto.blocks.basic import VectorEmbed
from teafacto.core.base import Val
from teafacto.util import argprun, ticktock, tokenize
import csv, numpy as np, sys


def readdata(trainp, testp, mode=None, masksym=-1, maxlen=100):
    assert(mode is not None)
    if mode is "char":
        return readdata_char(trainp, testp, maxlen=maxlen, masksym=masksym)
    elif mode is "word":
        return readdata_word(trainp, testp, maxlen=maxlen, masksym=masksym)


def readdata_word(trainp, testp, maxlen=100, masksym=-1):
    tt = ticktock("data reader")

    def readdataset(p, wdic, maxlen=100):
        dataret = []
        goldret = []
        toolong = 0
        realmaxlen = 0
        with open(p) as f:
            data = csv.reader(f, delimiter=",")
            for row in data:
                rowelems = tokenize(row[2])
                realmaxlen = max(realmaxlen, len(rowelems))
                if len(rowelems) > maxlen:
                    toolong += 1
                for rowelem in set(rowelems):
                    if rowelem not in wdic:
                        wdic[rowelem] = len(wdic)
                dataret.append([wdic[x] for x in rowelems])
                goldret.append(row[0])
        print "{} comments were too long".format(toolong)
        maxlen = min(maxlen, realmaxlen)
        datamat = np.ones((len(dataret) - 1, maxlen)).astype("int32") * masksym
        for i in range(1, len(dataret)):
            datamat[i - 1, :min(len(dataret[i]), maxlen)] = dataret[i][:min(len(dataret[i]), maxlen)]
        return datamat, np.asarray(goldret[1:], dtype="int32"), wdic

    tt.tick("reading data")
    traindata, traingold, wdic = readdataset(trainp, {}, maxlen=maxlen)
    testdata, testgold, wdic = readdataset(testp, wdic=wdic, maxlen=maxlen)
    tt.tock("data read")
    return (traindata, traingold), (testdata, testgold), wdic


def readdata_char(trainp, testp, maxlen=1000, masksym=-1):
    tt = ticktock("data reader")
    def readdataset(p):
        dataret = []
        goldret = []
        toolong = 0
        with open(p) as f:
            data = csv.reader(f, delimiter=",")
            for row in data:
                if len(row[2]) > maxlen:
                    toolong += 1
                dataret.append([ord(x) for x in row[2]])
                goldret.append(row[0])
        print "{} comments were too long".format(toolong)
        datamat = np.ones((len(dataret)-1, maxlen)).astype("int32") * masksym
        for i in range(1, len(dataret)):
            datamat[i-1, :min(len(dataret[i]), maxlen)] = dataret[i][:min(len(dataret[i]), maxlen)]
        return datamat, np.asarray(goldret[1:], dtype="int32")
    tt.tick("reading data")
    traindata, traingold = readdataset(trainp)
    testdata, testgold = readdataset(testp)
    allchars = set(list(np.unique(traindata))).union(set(list(np.unique(testdata))))
    chardic = dict(zip(list(allchars), range(len(allchars))))
    chardic[masksym] = masksym
    traindata = np.vectorize(lambda x: chardic[x])(traindata)
    testdata = np.vectorize(lambda x: chardic[x])(testdata)
    chardic = {chr(k): v for k, v in chardic.items() if k != masksym}
    tt.tock("data read")
    return (traindata, traingold), (testdata, testgold), chardic


def run(epochs=50,
        numbats=25,
        lr=0.1,
        layers=1,
        embdim=200,
        encdim=200,
        bidir=False,
        wordlevel=False,        # "char" or "word"
        maxlen=75,
        maxwordlen=15,
        ):
    maskid = -1
    charword = False
    mode = "word" if wordlevel else "char"
    (traindata, traingold), (testdata, testgold), dic = \
        readdata("../../../data/hatespeech/train.csv",
                 "../../../data/hatespeech/test.csv",
                 masksym=maskid, mode=mode, maxlen=maxlen)
    # data stats
    print "class distribution in train: {}% positive".format(np.sum(traingold)*1. / np.sum(np.ones_like(traingold)))
    print "class distribution in test: {}% positive".format(np.sum(testgold)*1. / np.sum(np.ones_like(testgold)))
    if mode == "word" and charword is True:      # create wordmat
        realmaxwordlen = reduce(lambda x, y: max(x,y), [len(z) for z in dic.keys()], 0)
        maxwordlen = min(realmaxwordlen, maxwordlen)
        wordmat = maskid * np.ones((len(dic), maxwordlen))
        chardic = {}
        revdic = {v: k for k, v in dic.items()}
        for i in range(wordmat.shape[0]):
            w = revdic[i]
            w = w[:min(len(w), maxwordlen)]
            for c in w:
                if c not in chardic:
                    chardic[c] = len(chardic)
            wordmat[i, :len(w)] = w

    inpemb = VectorEmbed(indim=len(dic), dim=embdim)
    if mode == "word" and charword:
        inpemb = SimpleSeq2Vec(indim=len(chardic), inpembdim=embdim,
                               innerdim=embdim, maskid=maskid)

    encdim = [encdim] * layers
    enc = SimpleSeq2Idx(inpemb=inpemb, inpembdim=embdim,
                        innerdim=encdim, maskid=maskid, bidir=bidir,
                        numclasses=2)
    #print enc.predict(traindata[:5, :])
    transf = lambda x: x
    if mode == "word" and charword:
        class PreProc(object):
            def __init__(self, wordmat):
                self.em = Val(wordmat)
            def __call__(self, x):
                return self.em[x]
        transf = PreProc(wordmat)
    enc = enc.train([traindata], traingold)\
        .adadelta(lr=lr).grad_total_norm(1.0)\
        .cross_entropy().split_validate(6, random=True).cross_entropy().accuracy()\
        .train(numbats=numbats, epochs=epochs)

    enc.save("hatemodel.{}.Emb{}D.Enc{}D.{}L.model".format(mode, embdim, encdim, layers))







if __name__ == "__main__":
    argprun(run)