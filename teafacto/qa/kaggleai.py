import os, re, pandas as pd, numpy as np
from IPython import embed
from teafacto.lm import Glove
from teafacto.encoders.rnnenc import *
from teafacto.core.rnn import *

def read(path="../../data/kaggleai/training_set.tsv"):
    path = os.path.join(os.path.dirname(__file__), path)
    file = open(path)
    todf = []
    first = True
    cols = None
    for line in file:
        ls = line[:-1].split("\t")
        if first:
            cols = ls
            first = False
            continue
        ls = [int(ls[0])] + [tokenize(x) for x in ls[1:]]
        todf.append(ls)
    df = pd.DataFrame(data=todf, columns=cols)
    df.index = df["id"]
    del df["id"]
    return df

def tokenize(ll):
    lls = ll.split(" ")
    if len(lls) == 1:
        if len(lls[0]) == 1:
            return lls[0]
    return map(lambda x: x.lower(), lls)

class StringTransformer(object):
    def __init__(self, wembs):
        self.wembs = wembs

    def questiontransformer(self, q): # q is a list of lowercase words
        return map(lambda x: self.wembs * x, q)

    def answertransformer(self, a): # a is a list of lowercase words
        try:
            return map(lambda x: self.wembs * x, a)
        except TypeError:
            print "TypeError"
            embed()

def transformdf(df, t):
    tdf = df.copy()
    tdf["question"] = tdf["question"].map(t.questiontransformer)
    tdf["correctAnswer"] = tdf["correctAnswer"].map(lambda x: {"A": 0, "B": 1, "C": 2, "D": 3}[x])
    tdf["answerA"] = tdf["answerA"].map(t.answertransformer)
    tdf["answerB"] = tdf["answerB"].map(t.answertransformer)
    tdf["answerC"] = tdf["answerC"].map(t.answertransformer)
    tdf["answerD"] = tdf["answerD"].map(t.answertransformer)
    return tdf


class QAEncDotSM(SMBase, Predictor, Saveable):
    def __init__(self, dim=50, innerdim=100, wreg=0.000001, wembs=Glove(50), **kw):
        super(QAEncDotSM, self).__init__(**kw)
        self.dim = dim
        self.innerdim = innerdim
        self.wreg = wreg
        self.qencoder = RNNEncoder() + GRU(dim=self.dim, innerdim=self.innerdim, wreg=self.wreg)
        self.aencoder = RNNEncoder() + GRU(dim=self.dim, innerdim=self.innerdim, wreg=self.wreg)
        self.wemb = wembs.theano

    def defmodel(self):
        iq, ia, ib, ic, id = T.imatrices("q", "a", "b", "c", "d") # (batsize, seqlen)
        q, a, b, c, d = [self.wemb[x, :] for x in [iq, ia, ib, ic, id]] # (batsize, seqlen, dim)
        embed()
        qenc = self.qencoder.encode(q) # qenc: (batsize, innerdim)
        aenc, benc, cenc, denc = [self.aencoder.encode(x) for x in [a, b, c, d]] # (batsize, innerdim)
        dots = [T.batched_dot(qenc, x).reshape((x.shape[0], 1)) for x in [aenc, benc, cenc, denc]]
        dots = T.concatenate(dots, axis=1) # (batsize, 4)
        probs = T.nnet.softmax(dots)
        golds = T.ivector("answers")
        return probs, golds, [q, a, b, c, d, golds]

    @property
    def depparameters(self):
        return self.qencoder.parameters + self.aencoder.parameters

    @property
    def ownparameters(self):
        return []

    def getsamplegen(self, data, labels):
        batsize = self.batsize
        self.sampleoffset = 0
        idxs = np.arange(0, data.shape[0], 1, dtype="int32")
        idxs = np.random.shuffle(idxs)
        def samplegen():
            # nonzeroidx = sorted(np.random.randint(0, trainX.shape[0], size=(batsize,)).astype("int32"))
            start = self.sampleoffset
            end = min(self.sampleoffset+ batsize, idxs.shape[0])
            selidxs = idxs[start:end]
            datasample = data[selidxs, :].astype("int32")
            labelsample = labels[selidxs].astype("int32")
            self.sampleoffset += end-start if end < idxs.shape[0] else 0
            return datasample, labelsample
        return samplegen



def run():
    df = read()
    glove = Glove(50)
    print glove * "word"
    print np.linalg.norm(glove.theano.get_value()[glove * "word", :] - glove % "words")
    tdf = transformdf(df, StringTransformer(glove))
    question_max_len = tdf["question"].map(lambda x: len(x)).max()
    answer_max_len = reduce(max, [tdf["answer%s" % x].map(lambda y: len(y)).max() for x in ["A", "B", "C", "D"]])

    wreg = 0.00001
    lr = 0.001
    epochs = 100
    numbats = 50

    model = QAEncDotSM(dim=50, innerdim=100, wreg=wreg, maxiter=epochs, numbats=numbats, validsplit=0.2).autosave \
            + SGD(lr=lr)

    traindata = tdf[["question", "answerA", "answerB", "answerC", "answerD"]].values
    labeldata = tdf["correctAnswer"].values
    err, verr = model.train(traindata, labeldata)

    embed()





if __name__ == "__main__":
    run()
