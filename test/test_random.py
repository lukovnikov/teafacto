from unittest import TestCase
from teafacto.core.base import RVal, Block, tensorops as T
from teafacto.blocks.basic import Dropout
from teafacto.blocks.activations import GumbelSoftmax
import numpy as np


class RandomSequence(Block):
    def __init__(self, **kw):
        super(RandomSequence, self).__init__(**kw)
        self.randval = RVal().normal((5,))

    def apply(self):
        out = T.scan(self.rec, sequences=None, outputs_info=[None], n_steps=5)
        return out

    def rec(self):
        return self.randval


class RandomSequenceInside(Block):
    def apply(self, x):
        out = T.scan(self.rec, sequences=x, outputs_info=[None])
        return out

    def rec(self, x_t):
        return RVal().normal(x_t.shape)


class DropoutSequence(Block):
    def __init__(self, **kw):
        super(DropoutSequence, self).__init__(**kw)
        self.dropout = Dropout(0.5, _alwaysrandom=True)

    def apply(self, x):
        out = T.scan(self.rec, sequences=x, outputs_info=[None])
        return out

    def rec(self, x_t):
        return self.dropout(x_t)


class GumbelSequence(Block):
    def __init__(self, shape=None, **kw):
        super(GumbelSequence, self).__init__(**kw)
        self.gumbel = GumbelSoftmax(temperature=0.3, shape=shape, _alwaysrandom=True)

    def apply(self, x):
        out = T.scan(self.rec, sequences=x, outputs_info=[None])
        return out

    def rec(self, x_t):     #(batsize, numclass)
        return self.gumbel(x_t)



class TestRandom(TestCase):
    def test_random_sequence(self):
        rs = RandomSequence()
        pred = rs.predict()
        print pred
        for i in range(pred.shape[1]-1):
            self.assertTrue(not np.allclose(pred[:, i], pred[:, i+1]))

    def test_random_sequence_inside(self):
        rs = RandomSequenceInside()
        d = np.random.random((5, 4, 3))
        pred = rs.predict(d)
        print pred
        for i in range(pred.shape[1]-1):
            self.assertTrue(not np.allclose(pred[:, i], pred[:, i+1]))
        self.assertEqual(d.shape, pred.shape)

    def test_dropout_sequence(self):
        m = DropoutSequence()
        d = np.ones((5, 20))
        pred = m.predict(d)
        print pred
        for i in range(pred.shape[1]-1):
            self.assertTrue(not np.allclose(pred[:, i], pred[:, i+1]))
        self.assertEqual(d.shape, pred.shape)

    def test_gumbel_sequence(self):
        shape = None # (4,3)
        m = GumbelSequence(shape=shape)
        d = np.ones((5, 4, 3))
        pred = m.predict(d)
        np.set_printoptions(precision=5, suppress=True)
        print pred
        self.assertEqual(d.shape, pred.shape)
        predsum = np.sum(pred, axis=-1)
        self.assertTrue(np.allclose(predsum, np.ones_like(predsum)))