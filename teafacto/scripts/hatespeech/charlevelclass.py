from teafacto.blocks.seq.enc import SimpleSeq2Idx
from teafacto.util import argprun, ticktock
import csv, numpy as np


def readdata(trainp, testp, maxlen=1000, masksym=-1):
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
    chardic[-1] = -1
    traindata = np.vectorize(lambda x: chardic[x])(traindata)
    testdata = np.vectorize(lambda x: chardic[x])(testdata)
    chardic = {chr(k): v for k, v in chardic.items() if k in range(256)}
    tt.tock("data read")
    return (traindata, traingold), (testdata, testgold), chardic


def run(epochs=50,
        numbats=100,
        lr=0.1,
        layers=1,
        embdim=100,
        encdim=200,
        bidir=False):
    maskid = -1
    (traindata, traingold), (testdata, testgold), chardic = readdata("../../../data/hatespeech/train.csv", "../../../data/hatespeech/test.csv", masksym=maskid)
    enc = SimpleSeq2Idx(indim=len(chardic), inpembdim=embdim,
                        innerdim=encdim, maskid=maskid, bidir=bidir,
                        layers=layers, numclasses=2)
    pred = enc.predict(traindata[:5, :])
    enc = enc.train([traindata], traingold).adadelta(lr=lr)\
        .cross_entropy()\
        .train(numbats=numbats, epochs=epochs)







if __name__ == "__main__":
    argprun(run)