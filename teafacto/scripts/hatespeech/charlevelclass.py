from teafacto.blocks.seq.enc import SimpleSeq2Idx
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

    def readdataset(p, wdic={}):
        dataret = []
        goldret = []
        toolong = 0
        with open(p) as f:
            data = csv.reader(f, delimiter=",")
            for row in data:
                rowelems = tokenize(row[2])
                if len(rowelems) > maxlen:
                    toolong += 1
                for rowelem in set(rowelems):
                    if rowelem not in wdic:
                        wdic[rowelem] = len(wdic)
                dataret.append([wdic[x] for x in rowelems])
                goldret.append(row[0])
        print "{} comments were too long".format(toolong)
        datamat = np.ones((len(dataret) - 1, maxlen)).astype("int32") * masksym
        for i in range(1, len(dataret)):
            datamat[i - 1, :min(len(dataret[i]), maxlen)] = dataret[i][:min(len(dataret[i]), maxlen)]
        return datamat, np.asarray(goldret[1:], dtype="int32"), wdic

    tt.tick("reading data")
    traindata, traingold, wdic = readdataset(trainp)
    testdata, testgold, wdic = readdataset(testp, wdic=wdic)
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
        mode="char",        # "char" or "word"
        maxlen=75,
        ):
    maskid = -1
    (traindata, traingold), (testdata, testgold), dic = \
        readdata("../../../data/hatespeech/train.csv",
                 "../../../data/hatespeech/test.csv",
                 masksym=maskid, mode=mode, maxlen=maxlen)
    enc = SimpleSeq2Idx(indim=len(dic), inpembdim=embdim,
                        innerdim=encdim, maskid=maskid, bidir=bidir,
                        layers=layers, numclasses=2)
    pred = enc.predict(traindata[:5, :])
    enc = enc.train([traindata], traingold).adadelta(lr=lr).grad_total_norm(1.0)\
        .cross_entropy().split_validate(6, random=True).accuracy()\
        .train(numbats=numbats, epochs=epochs)

    enc.save("hatemodel.{}.Emb{}D.Enc{}D.{}L.model".format(mode, embdim, encdim, layers))







if __name__ == "__main__":
    argprun(run)