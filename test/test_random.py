from unittest import TestCase
from teafacto.core.base import RVal, Block, tensorops as T
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


class TestRandom(TestCase):
    def test_random_sequence(self):
        rs = RandomSequence()
        pred = rs.predict()
        print pred
        for i in range(pred.shape[1]-1):
            self.assertTrue(not np.allclose(pred[:, i], pred[:, i+1]))