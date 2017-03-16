from unittest import TestCase
import numpy as np

from teafacto.blocks.loss import CrossEntropy
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

