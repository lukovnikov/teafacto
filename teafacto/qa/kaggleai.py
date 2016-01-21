import os, re, pandas as pd, numpy as np
from IPython import embed
from teafacto.lm import Glove
from teafacto.encoders.rnnenc import *
from teafacto.core.rnn import *
from teafacto.core.train import Trainer

from nltk.tokenize import RegexpTokenizer


# TODO GLOBAL:
# TODO: refactor: better management of dependencies in model expressions ==> explicit parameters and deps
#           getreg() of dep params from the dep class
#       ==> every block must have settings, params, deps, getreg(), ... --> blocks containing other blocks (containment trees)
#                   -> like this, contained blocks can be separately regularized (maybe also other things)

# TODO (Idea): refactor: separate Softmax/Ranking/objective functions from Base class
#       BUT: samplegen depends on objective
#       ==> use a composition pattern
#               defproblem() of Base class creates objective-specific problem definition
#                   calls defmodel for problem-specific expression
#               samplegen is overridden by specific problem expression definitions to match both objective and problem
#

# TODO HERE: check dims and stuff, because very slow

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
    tokens = RegexpTokenizer(r'\w+').tokenize(ll)
    if len(tokens) == 1 and len(tokens[0]) == 1: # one letter:
        return tokens[0]
    else:
        return map(lambda x: x.lower(), tokens)

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
    def __init__(self, dim=50, innerdim=100, wembs=None, **kw):
        super(QAEncDotSM, self).__init__(**kw)
        self.dim = dim
        self.innerdim = innerdim
        self.qencoder = RNNEncoder() + GRU(dim=self.dim, innerdim=self.innerdim, wreg=self.wreg)
        self.aencoder = RNNEncoder() + GRU(dim=self.dim, innerdim=self.innerdim, wreg=self.wreg)
        if wembs is None:
            wembs = Glove(50)
        self.wemb = wembs.theano

    def defmodel(self):
        iq, ia, ib, ic, id = T.imatrices("q", "a", "b", "c", "d") # (batsize, seqlen)
        q, a, b, c, d = [self.wemb[x, :] for x in [iq, ia, ib, ic, id]] # (batsize, seqlen, dim)
        qenc = self.qencoder.encode(q) # qenc: (batsize, innerdim)
        aenc, benc, cenc, denc = [self.aencoder.encode(x) for x in [a, b, c, d]] # (batsize, innerdim)
        dots = [T.batched_dot(qenc, x).reshape((x.shape[0], 1)) for x in [aenc, benc, cenc, denc]]
        #dots = [T.sqr((qenc - x).norm(2, axis=1)).reshape((x.shape[0], 1)) for x in [aenc, benc, cenc, denc]]
        dots = T.concatenate(dots, axis=1) # (batsize, 4)
        probs = T.nnet.softmax(dots)
        golds = T.ivector("answers")
        return probs, golds, [iq, ia, ib, ic, id, golds]

    @property
    def depparameters(self):
        return self.qencoder.parameters.union(self.aencoder.parameters)

    @property
    def ownparameters(self):
        return set() #{self.wemb}

    def getsamplegen(self, data, labels, onebatch=False): # data: ? list of (batsize, seqlen), seqlen for Q is different than for A's
        if onebatch:
            batsize = data.shape[0]
        else:
            batsize = self.batsize
        sampleoffsett = [0]
        idxs = np.arange(0, data.shape[0], 1, dtype="int32")
        np.random.shuffle(idxs)
        def samplegen():
            start = sampleoffsett[0]
            #embed()
            end = min(sampleoffsett[0] + batsize, idxs.shape[0])
            selidxs = idxs[start:end]
            datasample = data[selidxs, :].astype("int32")
            labelsample = labels[selidxs].astype("int32")
            sampleoffsett[0] += end-start if end < idxs.shape[0] else 0
            return datasample[:, 0, :], \
                   datasample[:, 1, :], \
                   datasample[:, 2, :], \
                   datasample[:, 3, :], \
                   datasample[:, 4, :], \
                   labelsample
        return samplegen



def run():

    wreg = 0.0
    epochs = 5
    numbats = 50
    lr = 10000. #0.001

    if False:
        df = read(path="../../data/kaggleai/test.tsv")
    else:
        df = read()
        df = df.iloc[0:100]
    glove = Glove(50)
    tdf = transformdf(df, StringTransformer(glove))

    trainer = Trainer(lambda:
            QAEncDotSM(dim=50, innerdim=100, wreg=wreg, maxiter=epochs, numbats=numbats, wembs=glove)
            + SGD(lr=lr)
    )

    qmat = pd.DataFrame(list(tdf["question"].values)).fillna(0).values.astype("int32")
    amats = [np.pad(y, ((0, 0), (0, qmat.shape[1] - y.shape[1])), mode="constant")
             for y in [pd.DataFrame(list(tdf[x].values)).fillna(0).values.astype("int32")
                       for x in ["answerA", "answerB", "answerC", "answerD"]]]
    traindata = np.stack([qmat] + amats, axis=1) # (numsam, 5, maxlen)
    labeldata = tdf["correctAnswer"].values
    #embed()
    models, err, verr, _, _, _ = trainer.train(traindata, labeldata, validsplit=5, validrandom=123, folds=5, average_err=False)
    model = models[0]

    embed()





if __name__ == "__main__":
    run()
