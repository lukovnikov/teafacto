from unittest import TestCase
from teafacto.core.base import Block, param, tensorops as T
import numpy as np


class TestLazyParam(TestCase):
    def test_lazy_param(self):
        class DummyBlock(Block):
            def __init__(self, **kw):
                super(DummyBlock, self).__init__(**kw)
                self.W = param((None, 12), name="test_param").uniform()

            def apply(self, x):
                self.W.shape[0] = x.kshape[1]
                a = T.dot(x, self.W)
                return a
        d = np.random.random((5, 4))
        b = DummyBlock()
        pred = b.predict(d)
        print pred

    def test_other_lazy_param(self):
        class DummyBlock(Block):
            def apply(self, x):
                self.W = param((x.kshape[1], 2), name="test_param").uniform()
                a = T.dot(x, self.W)
                a.kshape = (x.kshape[0], 2)
                return a
        d = np.random.random((5, 4))
        b = DummyBlock()
        pred = b.predict(d)
        print pred
