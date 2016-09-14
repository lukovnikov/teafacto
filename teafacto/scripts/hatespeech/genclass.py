from teafacto.blocks.seq.enc import SimpleSeq2Bool, SimpleSeq2Vec, SimpleSeq2Idx
from teafacto.blocks.seq.oldseqproc import SimpleSeqTransducer
from teafacto.blocks.basic import VectorEmbed, Eye
from teafacto.core.base import Val, Block, tensorops as T
from teafacto.util import argprun, ticktock, tokenize
import csv, numpy as np, sys
from IPython import embed


def readdata(trainp, testp, mode=None, maxlen=100):
    assert(mode is not None)
    if mode is "char":
        return readdata_char(trainp, testp, maxlen=maxlen)
    elif mode is "word":
        return readdata_word(trainp, testp, maxlen=maxlen)


def readdata_word(trainp, testp, maxlen=100):
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
        datamat = np.zeros((len(dataret) - 1, maxlen)).astype("int32")
        for i in range(1, len(dataret)):
            datamat[i - 1, :min(len(dataret[i]), maxlen)] = dataret[i][:min(len(dataret[i]), maxlen)]
        return datamat, np.asarray(goldret[1:], dtype="int32"), wdic

    tt.tick("reading data")
    traindata, traingold, wdic = readdataset(trainp, {"<PAD>": 0, "<START>": 1}, maxlen=maxlen)
    testdata, testgold, wdic = readdataset(testp, wdic=wdic, maxlen=maxlen)
    tt.tock("data read")
    return (traindata, traingold), (testdata, testgold), wdic


def readdata_char(trainp, testp, maxlen=1000):
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
        datamat = -np.ones((len(dataret)-1, maxlen)).astype("int32")
        for i in range(1, len(dataret)):
            datamat[i-1, :min(len(dataret[i]), maxlen)] = dataret[i][:min(len(dataret[i]), maxlen)]
        return datamat, np.asarray(goldret[1:], dtype="int32")
    tt.tick("reading data")
    traindata, traingold = readdataset(trainp)
    testdata, testgold = readdataset(testp)
    allchars = set(list(np.unique(traindata))).union(set(list(np.unique(testdata))))
    chardic = {-1: 0}
    allchars.remove(-1)
    chardic.update(dict(zip(list(allchars), range(2, len(allchars) + 2))))
    traindata = np.vectorize(lambda x: chardic[x])(traindata)
    testdata = np.vectorize(lambda x: chardic[x])(testdata)
    del chardic[-1]
    chardic = {chr(k): v for k, v in chardic.items()}
    chardic["<PAD>"] = 0
    chardic["<START>"] = 1
    tt.tock("data read")
    return (traindata, traingold), (testdata, testgold), chardic


class GenClass(Block):
    def __init__(self, symemb, classemb, transducer, **kw):
        super(GenClass, self).__init__(**kw)
        self.transducer = transducer
        self.wemb = symemb
        self.cemb = classemb

    def apply(self, seq, clas):     # seq: idx^(batsize, seqlen), clas: idx^(batsize,)
        seqemb = self.wemb(seq)     # (batsize, seqlen, wembdim)
        clasemb = self.cemb(clas)   # (batsize, cembdim)
        clasemb = clasemb.dimshuffle(0, 'x', 1).repeat(seqemb.shape[1], axis=1)
        ret = T.concatenate([seqemb, clasemb], axis=2)
        return self.transducer(ret)


def run(epochs=50,
        numbats=25,
        lr=0.1,
        layers=1,
        embdim=100,
        encdim=200,
        bidir=False,
        wordlevel=False,        # "char" or "word"
        maxlen=75,
        maxwordlen=15,
        ):
    mode = "word" if wordlevel else "char"
    (traindata, traingold), (testdata, testgold), dic = \
        readdata("../../../data/hatespeech/train.csv",
                 "../../../data/hatespeech/test.csv",
                 mode=mode, maxlen=maxlen)

    revdic = {v: k for k, v in dic.items()}
    def pp(s):
        print "".join([revdic[x] if x in revdic else "<???>" for x in s])

    embed()
    # data stats
    print "class distribution in train: {}% positive".format(np.sum(traingold)*1. / np.sum(np.ones_like(traingold)))
    print "class distribution in test: {}% positive".format(np.sum(testgold)*1. / np.sum(np.ones_like(testgold)))

    wordemb = VectorEmbed(indim=len(dic), dim=embdim)
    clasemb = VectorEmbed(indim=2, dim=embdim)
    encdim = [encdim] * layers
    enc = SimpleSeqTransducer(inpemb=Eye(embdim*2), innerdim=encdim,
                              outdim=len(dic))

    m = GenClass(wordemb, clasemb, enc)


    # shift traindata
    straindata = np.zeros((traindata.shape[0], 1), dtype="int32")
    straindata = np.concatenate([straindata, traindata[:, :-1]], axis=1)

    m = m.train([straindata, traingold], traindata)\
        .adadelta(lr=lr).grad_total_norm(1.0).seq_cross_entropy()\
        .split_validate(6, random=True).seq_cross_entropy().seq_accuracy()\
        .train(numbats=numbats, epochs=epochs)

    #enc.save("hatemodel.{}.Emb{}D.Enc{}D.{}L.model".format(mode, embdim, encdim, layers))


    # pre predict
    stestdata = np.zeros((testdata.shape[0], 1), dtype="int32")
    stestdata = np.concatenate([stestdata, testdata[:, :-1]], axis=1)
    negpreds = m.predict(stestdata, np.zeros_like(testgold))  # (batsize, seqlen, vocsize)
    pospreds = m.predict(stestdata, np.ones_like(testgold))
    negprobs = negpreds[
        np.arange(negpreds.shape[0])[:, None],
        np.arange(negpreds.shape[1])[None, :],
        testdata]
    posprobs = pospreds[
        np.arange(pospreds.shape[0])[:, None],
        np.arange(pospreds.shape[1])[None, :],
        testdata]
    negprobs = np.sum(-np.log(negprobs), axis=1)
    posprobs = np.sum(-np.log(posprobs), axis=1)
    pred = negprobs < posprobs
    embed()





if __name__ == "__main__":
    argprun(run)