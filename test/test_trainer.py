from unittest import TestCase

import numpy as np

from teafacto.examples.dummy import *

'''
    pred = ae.predict(pdata)
    print pred.shape
    print np.argmax(pred, axis=1)
    #print err, verr
'''

class TestTrainingPause(TestCase):
    def test_second_training_cycle(self):
        dim = 50
        epochs1 = 3
        epochs2 = 3
        self.vocabsize = 2000
        normalize = True
        ae = Dummy(indim=self.vocabsize, dim=dim, normalize=normalize)
        aefresh = Dummy.unfreeze(ae.freeze())
        frozen, self.err1 = self.trainfreeze(ae, epochs1)
        ae = Dummy.unfreeze(frozen)
        frozen, self.err2 = self.trainfreeze(ae, epochs2)
        frozen, self.err1p2 = self.trainfreeze(aefresh, epochs1+epochs2)
        self.assertLess(self.err2, self.err1)
        self.assertTrue(np.allclose(self.err1p2, self.err2))

    def trainfreeze(self, ae, epochs):
        numbats = 100
        lr = 0.2
        data = np.arange(0, self.vocabsize).astype("int32")
        self.err, self.verr, _, _ = ae.train([data], data).adadelta(lr=lr).neg_log_prob().train(numbats=numbats, epochs=epochs)
        frozen = ae.freeze()
        err = self.err[-1]
        return frozen, err


class TestModelTrainer(TestCase):
    def setUp(self):
        dim=50
        self.epochs=4
        self.vocabsize=2000
        self.lrthresh = 2
        normalize=True
        self.ae = Dummy(indim=self.vocabsize, dim=dim, normalize=normalize)
        self.train()

    def train(self):
        numbats=100
        lr=0.02
        lr *= numbats
        data = np.arange(0, self.vocabsize).astype("int32")
        self.err, self.verr, _, _ = \
            self.ae.train([data], data).adadelta(lr=lr).dlr_thresh(thresh=self.lrthresh).neg_log_prob() \
                .autovalidate().neg_log_prob().accuracy()\
            .train(numbats=numbats, epochs=self.epochs)

    def test_embeddings_normalized(self):
        pdata = range(self.vocabsize)
        pembs = self.ae.W.predict(pdata)
        norms = np.linalg.norm(pembs, axis=1)
        expectednorms = np.ones((self.vocabsize,))
        self.assertTrue(np.allclose(norms, expectednorms))

    def test_adaptive_learning_rate(self):
        differrs = self.err[:self.lrthresh]
        sameerrs = self.err[self.lrthresh:]
        for i in range(len(differrs)-1):
            for j in range(i+1, len(differrs)):
                self.assertFalse(np.allclose(differrs[i], differrs[j]))
        for i in range(len(sameerrs)):
            for j in range(len(sameerrs)):
                self.assertTrue(np.allclose(sameerrs[i], sameerrs[j]))


class TestModelTrainerNovalidate(TestModelTrainer):

    def train(self):
        numbats=100
        lr=0.02
        lr *= numbats
        data = np.arange(0, self.vocabsize).astype("int32")
        self.err, self.verr, _, _ = \
            self.ae.train([data], data).adadelta(lr=lr).dlr_thresh(thresh=self.lrthresh).neg_log_prob() \
            .train(numbats=numbats, epochs=self.epochs)

    def test_embeddings_normalized(self):
        pass


class TestModelTrainerValidsplit(TestModelTrainerNovalidate):
    def train(self):
        numbats=100
        lr=0.02
        lr *= numbats
        data = np.arange(0, self.vocabsize).astype("int32")
        self.err, self.verr, _, _ = \
            self.ae.train([data], data).adadelta(lr=lr).dlr_thresh(thresh=self.lrthresh).neg_log_prob() \
            .split_validate(5, random=True).neg_log_prob() \
            .train(numbats=numbats, epochs=self.epochs)


class TestModelTrainerCrossValid(TestModelTrainerNovalidate):
    def train(self):
        numbats=100
        lr=0.02
        lr *= numbats
        data = np.arange(0, self.vocabsize).astype("int32")
        self.err, self.verr, _, _ = \
            self.ae.train([data], data).adadelta(lr=lr).dlr_thresh(thresh=self.lrthresh).neg_log_prob() \
            .cross_validate(5, random=True).neg_log_prob() \
            .train(numbats=numbats, epochs=self.epochs)


class TestModelTrainerAutovalidate(TestModelTrainerNovalidate):
    def train(self):
        numbats = 100
        lr=0.02
        lr *= numbats
        data = np.arange(0, self.vocabsize).astype("int32")
        self.err, self.verr, _, _ = \
            self.ae.train([data], data).adadelta(lr=lr).dlr_thresh(thresh=self.lrthresh).neg_log_prob() \
            .autovalidate(splits=5, random=True).neg_log_prob() \
            .train(numbats=numbats, epochs=self.epochs)


