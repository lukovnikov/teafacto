import theano
from math import ceil
import numpy as np
from teafacto.core.utils import ticktock as TT


class SplitIdxIterator(object):
    def __init__(self, datalen, split=10, random=False, folds=1):
        self.folds = folds
        self.splits = self.buildsplits(datalen, random, split, folds)

    def buildsplits(self, datalen, random, split, folds):    # random: whether and how random, split: percentage in split, folds: how many times to produce a split
        dataidxs = np.arange(0, datalen, 1, dtype="int32")
        if random is not False:     # do random splitting but not Monte Carlo
            if isinstance(random, (int, long)):  # set seed
                np.random.seed(random)
            np.random.shuffle(dataidxs)
        # generate a list of vectors of data indexes
        offset = 0
        splitsize = int(ceil(1. * datalen / split))
        currentfold = 0
        splits = []
        while currentfold < folds:
            start = offset
            end = min(offset + splitsize, datalen)
            splitidxs = dataidxs[start:end]
            splits.append(splitidxs)
            if end == datalen:  # restart
                if random is not False:     # reshuffle
                    np.random.shuffle(dataidxs)
                offset = 0
            currentfold += 1
            offset += splitsize
        return splits

    def __iter__(self):
        self.currentfold = 0
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.currentfold < self.folds:
            ret = self.splits[self.currentfold]       # get the indexes
            self.currentfold += 1
            return ret
        else:
            raise StopIteration


class Trainer(object):
    def __init__(self, modelbuilder, **params):
        self.modelbuilder = modelbuilder
        self.params = params
        self.tt = TT("Trainer")

    def train(self, data, labels, validsplit=10, validrandom=False, folds=1, validinter=1, tester=x, average_err=True):
        assert data.shape[0] == labels.shape[0]
        self.validsplit = validsplit
        self.tt.tick("training")
        if self.validsplit > 1:  # do validation during training
            self.splitter = SplitIdxIterator(data.shape[0], split=validsplit, random=validrandom, folds=folds)
            err = []
            verr = []
            testres = []
            models = []
            c = 0
            for splitidxs in self.splitter:
                tdata, tlabels, vdata, vlabels = self.splitdata(data, labels, splitidxs)
                m = self.modelbuilder()
                serr, sverr = m.train(tdata, tlabels, vdata, vlabels, evalinter=validinter, average_err=average_err)
                models.append(m)
                err.append(serr)
                verr.append(sverr)
                if tester is not None:
                    testre = tester.run(m, vdata, vlabels)
                    testres.append(testre)
                c += 1
                self.tt.progress(c, len(self.splitter.splits))
            err = np.asarray(err)
            avgerr = np.mean(err, axis=0)
            verr = np.asarray(verr)
            avgverr = np.mean(verr, axis=0)
            self.models = models
            self.tt.tock("done")
            return models, avgerr, avgverr, testres, err, verr
        else:
            m = self.modelbuilder()
            err = m.train(data, labels)
            verr = []
            return [m], err, verr, None, None, None

    def predict(self, data, model):
        model.predict(data)

    def splitdata(self, data, labels, splitidxs):
        validdata = data[splitidxs, :]
        validlabels = labels[splitidxs]
        traindata = np.delete(data, splitidxs, axis=0)
        trainlabels = np.delete(labels, splitidxs)
        return traindata, trainlabels, validdata, validlabels