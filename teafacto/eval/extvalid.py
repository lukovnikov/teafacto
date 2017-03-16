from teafacto.core.trainer import ExternalObjective
from teafacto.core.base import asblock
from IPython import embed
import numpy as np


class Perplexity(ExternalObjective):
    def __init__(self, model):
        super(Perplexity, self).__init__()
        self.predf = model.predict
        self.maskf = asblock(lambda *x: model(*x).mask).predict
        self.aggmode = "sum"

    def __call__(self, *sampleinps):
        gold = sampleinps[-1]
        inps = sampleinps[:-1]
        pred = self.predf(*inps)
        predmask = self.maskf(*inps)
        flatpred = pred.reshape(-1, pred.shape[-1])
        flatmask = predmask.flatten()
        flatgold = gold.flatten()
        flatce = -np.log(flatpred[np.arange(0, len(flatpred)), flatgold])
        flatce *= flatmask
        ce = flatce
        #ce = -np.log(pred[np.arange(0, len(pred)), gold[np.arange(0, len(pred))]])
        #ce = ce * predmask[..., np.newaxis]
        self.current_agg_error += np.sum(ce)
        self.current_agg_norma += np.sum(predmask)
        return 2 ** (np.sum(ce) / np.sum(predmask))

    def get_agg_error(self):
        if self.current_agg_norma == 0:
            return -0.
        return 2 ** (self.current_agg_error / self.current_agg_norma)

    def update_agg(self, err, numex):
        pass


class Accuracy(ExternalObjective):
    def __init__(self, model):
        super(Accuracy, self).__init__()
        self.predf = model.predict

    def __call__(self, *sampleinps):
        gold = sampleinps[-1]
        inps = sampleinps[:-1]
        pred = self.predf(*inps)
        ret = np.sum(np.argmax(pred, axis=1) == gold)
        return ret * 1. / len(gold)
