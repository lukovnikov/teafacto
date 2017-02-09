from unittest import TestCase
from teafacto.core.base import Block, param, tensorops as T
import numpy as np


class DummyBlock(Block):
    def __init__(self, **kw):
        super(DummyBlock, self).__init__(**kw)
        self.W = param((5, 12), name="test_param").uniform()

    def apply(self, x):
        a = T.dot(x, self.W)
        b = T.sum(a, axis=1)
        c = T.sum(b, axis=0)
        a.printas("a")
        a.push_extra_outs({"b": b, "c": c})
        return a


class TestExtraRet(TestCase):
    def test_extra_ret(self):
        b = DummyBlock()
        d = np.random.random((10, 5)).astype("float32")
        pred, extra = b.predict(d, _extra_outs=True)
        print pred.shape
        print extra
        self.assertTrue("b" in extra)
        self.assertEqual(extra["b"].shape, (10,))
        self.assertTrue("c" in extra)

    def test_extra_ret_selective(self):
        b = DummyBlock()
        d = np.random.random((10, 5)).astype("float32")
        pred, extra = b.predict(d, _extra_outs=["b"])
        self.assertIn("b", extra)
        self.assertNotIn("c", extra)

    def test_extra_ret_through_scan(self):
        class DummyRecBlock(Block):
            def apply(self, x):
                outs = T.scan(fn=self.f, sequences=[x])
                return outs
            def f(self, s):
                a = s * 2
                b = T.sum(a, axis=0)
                c = T.sum(a, axis=0)
                a.push_extra_outs({"b": b, "c": c})
                return a
        b = DummyRecBlock()
        d = np.random.random((3, 10, 5)).astype("float32")
        pred, extra = b.predict(d, _extra_outs=True)
        self.assertIn("b", extra)
        self.assertIn("c", extra)

