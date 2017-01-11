from unittest import TestCase
from teafacto.blocks.match import BilinearDistance, LinearDistance, CosineDistance, DotDistance
import numpy as np


class TestBilinearDistance(TestCase):
    def test_shape_normal(self):
        batsize = 10
        ldim = 5
        rdim = 4
        l = np.random.random((batsize, ldim))
        r = np.random.random((batsize, rdim))
        b = BilinearDistance(ldim, rdim)
        b.W.value.set_value(np.ones((rdim, ldim), dtype="float32"))
        pred = b.predict(l, r)
        self.assertEqual(pred.shape, (batsize,))

    def test_seq(self):
        batsize = 1
        ldim = 3
        rdim = 2
        seqlen = 4
        l = np.random.random((batsize, ldim))
        r = np.random.random((batsize, seqlen, rdim))
        b = BilinearDistance(ldim, rdim)
        pred = b.predict(l, r)
        l = l[0]
        r = r[0]
        for i in range(seqlen):
            x = np.dot(r[i], b.W.value.get_value())
            x = np.dot(x, l)
            self.assertEqual(x, pred[0][i])


class TestCosineDistance(TestCase):
    def test_shape(self):
        batsize = 10
        seqlen=3
        ldim = 5
        rdim = 5
        l = np.random.random((batsize, ldim))
        r = np.random.random((batsize, seqlen, rdim))
        b = CosineDistance()
        pred, extra = b.predict(l, r, _extra_outs=["lnorms", "rnorms"])
        print extra
        print pred
        self.assertEqual(pred.shape, (batsize, seqlen))
        self.assertTrue(np.all((pred - np.ones_like(pred)) < 0))


class TestDotDistance(TestCase):
    def test_shape(self):
        batsize = 10
        ldim = 5
        rdim = 5
        l = np.random.random((batsize, ldim))
        r = np.random.random((batsize, rdim))
        b = DotDistance()
        pred = b.predict(l, r)
        self.assertEqual(pred.shape, (batsize,))


class TestLinearDistance(TestCase):
    def test_shape(self):
        batsize = 10
        ldim = 5
        rdim = 4
        aggdim = 7
        l = np.random.random((batsize, ldim))
        r = np.random.random((batsize, rdim))
        b = LinearDistance(ldim, rdim, aggdim)
        pred = b.predict(l, r)
        self.assertEqual(pred.shape, (batsize,))

    def test_get_params(self):
        d = LinearDistance(10, 10, 10)
        params = {d.lin.W, d.lin.b, d.lin2.W, d.lin2.b, d.agg}
        self.assertEqual(params, d.get_params())

    def test_seq(self):
        batsize = 1
        ldim = 3
        rdim = 2
        seqlen = 4
        np.random.seed(544)
        l = np.random.random((batsize, ldim))
        r = np.random.random((batsize, seqlen, rdim))
        b = LinearDistance(ldim, rdim, 5)
        pred = b.predict(l, r)
        l = l[0]
        r = r[0]
        for i in range(seqlen):
            x = np.dot(l, b.lin.W.value.get_value()) + b.lin.b.value.get_value()
            y = np.dot(r[i], b.lin2.W.value.get_value()) + b.lin2.b.value.get_value()
            z = np.dot(x + y, b.agg.value.get_value())
            self.assertTrue(np.isclose(z, pred[0][i]))
