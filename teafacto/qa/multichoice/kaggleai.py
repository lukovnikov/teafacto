import pandas as pd
from nltk.tokenize import RegexpTokenizer

from teafacto.core.train import Trainer
from teafacto.core.utils import argparsify
from teafacto.encoders.rnnenc import *
from teafacto.eval.eval import ClasEvaluator
from teafacto.eval.metrics import ClassAccuracy
from teafacto.lm import Glove

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
from teafacto.qa.multichoice.models import *


def read(path="../../../data/kaggleai/training_set.tsv"):
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


def run(
        wreg=0.0,
        epochs=1,
        numbats=50,
        lr=10e-20,
        dims=50
    ):
    if False:
        df = read(path="../../../data/kaggleai/test.tsv")
    else:
        df = read()
    glove = Glove(dims)
    tdf = transformdf(df, StringTransformer(glove))

    trainer = Trainer(lambda:
                      RNNMaskedSumEncDotSM(dim=dims, wreg=wreg, maxiter=epochs, numbats=numbats, wembs=glove)
                      + SGD(lr=lr)
    )

    qmat = pd.DataFrame(list(tdf["question"].values)).fillna(0).values.astype("int32")
    amats = [np.pad(y, ((0, 0), (0, qmat.shape[1] - y.shape[1])), mode="constant")
             for y in [pd.DataFrame(list(tdf[x].values)).fillna(0).values.astype("int32")
                       for x in ["answerA", "answerB", "answerC", "answerD"]]]
    traindata = np.stack([qmat] + amats, axis=1) # (numsam, 5, maxlen)
    labeldata = tdf["correctAnswer"].values
    #embed()
    evaluator = ClasEvaluator(ClassAccuracy())
    models, err, verr, tres, _, _ = trainer.train(traindata, labeldata, validsplit=5, validrandom=123, folds=5, tester=evaluator)
    model = models[0]
    print model.wemb.get_value()[[0, 1, 2, 50], :]
    print tres

    #embed()


if __name__ == "__main__":
    run(**argparsify(run))
