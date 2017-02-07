from teafacto.blocks.seq.enc import SimpleSeq2Bool, SimpleSeq2Vec, SimpleSeq2Idx
from teafacto.blocks.lang.sentenc import WordCharSentEnc
from teafacto.blocks.seq.rnn import EncLastDim
from teafacto.blocks.basic import VectorEmbed, SMOWrap
from teafacto.core.base import Val
from teafacto.util import argprun, ticktock, tokenize
from teafacto.procutil import wordmat2chartensor
import csv, numpy as np, sys


def readdata(trainp, testp, mode=None, masksym=-1, maxlen=100):
    assert(mode is not None)
    if mode is "char":
        return readdata_char(trainp, testp, maxlen=maxlen, masksym=masksym)
    elif mode is "word":
        return readdata_word(trainp, testp, maxlen=maxlen, masksym=masksym)
    elif mode is "wordchar":
        (traindata, traingold), (testdata, testgold), dic = readdata_word(trainp, testp, maxlen=maxlen, masksym=masksym)
        traindata = wordmat2chartensor(traindata, dic)
        testdata = wordmat2chartensor(testdata, dic)

        allchars = set(list(np.unique(traindata))).union(set(list(np.unique(testdata))))
        allchars.remove(masksym)
        chardic = dict(zip(allchars, range(len(allchars))))
        chardic[masksym] = masksym
        chartrans = np.vectorize(lambda x: chardic[x])
        traindata = chartrans(traindata)
        testdata = chartrans(testdata)
        del chardic[masksym]
        chardic = {chr(k): v for k, v in chardic.items()}
        return (traindata, traingold), (testdata, testgold), chardic


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
        embdim=100,
        encdim=200,
        bidir=False,
        mode="wordchar",        # "char" or "word" or "wordchar"
        maxlen=75,
        maxwordlen=15,
        ):
    maskid = -1
    (traindata, traingold), (testdata, testgold), dic = \
        readdata("../../../data/hatespeech/train.csv",
                 "../../../data/hatespeech/test.csv",
                 masksym=maskid, mode=mode, maxlen=maxlen)

    # data stats
    print "class distribution in train: {}% positive".format(np.sum(traingold)*1. / np.sum(np.ones_like(traingold)))
    print "class distribution in test: {}% positive".format(np.sum(testgold)*1. / np.sum(np.ones_like(testgold)))

    inpemb = VectorEmbed(indim=len(dic), dim=embdim)
    encdim = [encdim] * layers
    if mode == "wordchar":
        enc = WordCharSentEnc(charemb=inpemb, charinnerdim=embdim,
                              wordemb=False, wordinnerdim=encdim,
                              maskid=maskid, bidir=bidir)
    else:
        enc = SimpleSeq2Vec(inpemb=inpemb, innerdim=encdim, maskid=maskid, bidir=bidir)

    m = SMOWrap(enc, outdim=2, nobias=True)
    #print enc.predict(traindata[:5, :])
    m = m.train([traindata], traingold)\
        .adadelta(lr=lr).grad_total_norm(1.0)\
        .cross_entropy().split_validate(6, random=True).cross_entropy().accuracy()\
        .train(numbats=numbats, epochs=epochs)

    m.save("hatemodel.{}.Emb{}D.Enc{}D.{}L.model".format(mode, embdim, encdim, layers))







if __name__ == "__main__":
    argprun(run)