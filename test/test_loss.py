from unittest import TestCase
import numpy as np

from teafacto.blocks.loss import CrossEntropy
from teafacto.blocks.basic import Softmax
from teafacto.core import Val


class TestCrossEntropy(TestCase):
    def test_1D(self):
        ce = CrossEntropy()
        batsize, vocsize = 10, 20
        datad = np.random.random((batsize, vocsize))
        goldd = np.random.randint(0, vocsize, (batsize,))
        data = Val(datad)
        gold = Val(goldd)
        pred = ce(data, gold)
        predpred = pred.eval()
        self.assertEqual(predpred.shape, (batsize,))
        self.assertTrue(np.allclose(predpred, -np.log(datad[np.arange(0, goldd.shape[0]), goldd])))

    def test_2D(self):
        ce = CrossEntropy()
        batsize, seqlen, vocsize = 10, 5, 20
        datad = np.random.random((batsize, seqlen, vocsize))
        goldd = np.random.randint(0, vocsize, (batsize, seqlen))
        data = Val(datad)
        gold = Val(goldd)
        pred = ce(data, gold)
        predpred = pred.eval()
        self.assertEqual(predpred.shape, (batsize,))
        for i in range(len(goldd)):
            x = np.sum(-np.log(datad[i][np.arange(0, seqlen), goldd[i]]))
            self.assertTrue(np.allclose(x, predpred[i]))

    def test_2D_masked(self):
        ce = CrossEntropy()
        batsize, seqlen, vocsize = 10, 5, 20
        datad = np.random.random((batsize, seqlen, vocsize))
        goldd = np.random.randint(0, vocsize, (batsize, seqlen))
        maskd = np.random.random((batsize, seqlen)) > 0.3
        data = Val(datad)
        gold = Val(goldd)
        mask = Val(maskd)
        data.mask = mask
        pred = ce(data, gold)
        predpred = pred.eval()
        self.assertEqual(predpred.shape, (batsize,))
        for i in range(len(goldd)):
            x = np.sum(-np.log(datad[i][np.arange(0, seqlen), goldd[i]]) * maskd[i])
            self.assertTrue(np.allclose(x, predpred[i]))

    def test_2D_allmean(self):
        ce = CrossEntropy(mode="allmean")
        batsize, seqlen, vocsize = 10, 5, 20
        datad = np.random.random((batsize, seqlen, vocsize))
        goldd = np.random.randint(0, vocsize, (batsize, seqlen))
        data = Softmax()(Val(datad))
        gold = Val(goldd)
        pred = ce(data, gold)
        predpred = pred.eval()
        x = -np.log((1./vocsize))
        print x
        datad = data.eval()

        self.assertEqual(predpred.shape, (batsize,))
        acc = 0
        for i in range(len(goldd)):
            x = np.sum(-np.log(datad[i][np.arange(0, seqlen), goldd[i]]))
            acc += x
        y = acc / (datad.shape[0] * datad.shape[1])
        self.assertTrue(np.allclose(predpred, y))

    def test_2D_allmean_masked(self):
        ce = CrossEntropy(mode="allmean")
        batsize, seqlen, vocsize = 10, 5, 20
        datad = np.random.random((batsize, seqlen, vocsize))
        goldd = np.random.randint(0, vocsize, (batsize, seqlen))
        maskd = np.random.random((batsize, seqlen)) > 0.3
        data = Val(datad)
        data.mask = Val(maskd)
        data = Softmax()(data)
        gold = Val(goldd)
        pred = ce(data, gold)
        predpred = pred.eval()
        x = -np.log((1./vocsize))
        print x
        datad = data.eval()

        self.assertEqual(predpred.shape, (batsize,))
        acc = 0
        for i in range(len(goldd)):
            x = np.sum(-np.log(datad[i][np.arange(0, seqlen), goldd[i]]) * maskd[i])
            acc += x
        y = acc / maskd.sum()
        self.assertTrue(np.allclose(predpred, y))

    def test_2D_allmean_masked_big(self):
        ce = CrossEntropy(mode="allmean")
        batsize, seqlen, vocsize = 10, 50, 1e5
        datad = np.random.random((batsize, seqlen, vocsize))
        goldd = np.random.randint(0, vocsize, (batsize, seqlen))
        maskd = np.random.random((batsize, seqlen)) > 0.6
        data = Val(datad)
        data.mask = Val(maskd)
        data = Softmax()(data)
        gold = Val(goldd)
        pred = ce(data, gold)
        predpred = pred.eval()
        x = -np.log((1./vocsize))
        print x
        print predpred
        datad = data.eval()

        self.assertEqual(predpred.shape, (batsize,))
        acc = 0
        for i in range(len(goldd)):
            x = np.sum(-np.log(datad[i][np.arange(0, seqlen), goldd[i]]) * maskd[i])
            acc += x
        y = acc / maskd.sum()
        self.assertTrue(np.allclose(predpred, y))



