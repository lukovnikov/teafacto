import pandas as pd

from teafacto.core.train import Trainer
from teafacto.core.utils import argparsify
from teafacto.encoders.rnnenc import *
from teafacto.eval.eval import ClasEvaluator
from teafacto.eval.metrics import ClassAccuracy
from teafacto.lm import Glove
from teafacto.qa.multichoice.models import *

from teafacto.qa.multichoice.kaggleutils import *


def _getpathfromhere(p):
    return os.path.join(os.path.dirname(__file__), p)

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
    tdf["answerA"] = tdf["answerA"].map(t.answertransformer)
    tdf["answerB"] = tdf["answerB"].map(t.answertransformer)
    tdf["answerC"] = tdf["answerC"].map(t.answertransformer)
    tdf["answerD"] = tdf["answerD"].map(t.answertransformer)
    if "correctAnswer" in tdf:
        tdf["correctAnswer"] = tdf["correctAnswer"].map(lambda x: {"A": 0, "B": 1, "C": 2, "D": 3}[x])
    return tdf


def run(
        wreg=0.0,
        epochs=0,
        numbats=50,
        lr=0.0,
        dims=50,
        predicton="../../../data/kaggleai/validation_set.tsv"
    ):
    def innerrun():
        glove = Glove(dims)
        trainer = Trainer(lambda:
                          DotSumEncSM(dim=dims, wreg=wreg, maxiter=epochs, numbats=numbats, wembs=glove)
                          + SGD(lr=lr)
        )
        return trainer, glove
    def innerdata(datap=None, wembs=None):
        if wembs is None:
            wembs = Glove(dims)
        df = read() if datap is None else read(datap)
        tdf = transformdf(df, StringTransformer(wembs))
        qmat = pd.DataFrame(list(tdf["question"].values)).fillna(0).values.astype("int32")
        amats = [np.pad(y, ((0, 0), (0, qmat.shape[1] - y.shape[1])), mode="constant")
                 for y in [pd.DataFrame(list(tdf[x].values)).fillna(0).values.astype("int32")
                           for x in ["answerA", "answerB", "answerC", "answerD"]]]
        traindata = np.stack([qmat] + amats, axis=1) # (numsam, 5, maxlen)
        if "correctAnswer" in tdf:
            labeldata = tdf["correctAnswer"].values
        else:
            labeldata = tdf
            #embed()
        return traindata, labeldata
    if predicton is None: # training mode
        trainer, glove = innerrun()
        traindata, labeldata = innerdata(wembs=glove)
        evaluator = ClasEvaluator(ClassAccuracy())
        models, err, verr, tres, _, _ = trainer.train(traindata, labeldata, validsplit=5, validrandom=123, folds=5, tester=evaluator)
        #model = models[0]
        #print model.wemb.get_value()[[0, 1, 2, 50], :]
        print tres
    else: # prediction mode
        trainer, glove = innerrun()
        traindata, labeldata = innerdata(wembs=glove)
        models, err, _, _, _, _ = trainer.train(traindata, labeldata, validsplit=1) # train on everything
        model = models[0]
        preddata, origdf = innerdata(wembs=glove, datap=predicton) # no labels
        predictions = model.predict(preddata)
        retdf = pd.DataFrame()
        retdf["id"] = origdf.index
        retdf["correctAnswer"] = map(lambda x: str(unichr(x+65)), predictions)
        retdf.set_index(["id"], inplace=True)
        retdf.to_csv(_getpathfromhere(predicton+".out.csv"))
    #embed()


if __name__ == "__main__":
    run(**argparsify(run))













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