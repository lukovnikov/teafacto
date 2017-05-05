from unittest import TestCase
from teafacto.blocks.match import *
import numpy as np


class TestEuclideanDistance(TestCase):
    def test_seq(self):
        batsize = 10
        ldim = 5
        rdim = 5
        seqlen = 6
        l = np.random.random((batsize, ldim))
        r = np.random.random((batsize, seqlen, rdim))
        b = EuclideanDistance()
        pred = b.predict(l, r)
        print pred.shape
        self.assertEqual(pred.shape, (batsize, seqlen))

    def test_seq_gated(self):
        batsize = 10
        ldim = 5
        rdim = 5
        seqlen = 6
        l = np.random.random((batsize, ldim))
        r = np.random.random((batsize, seqlen, rdim))
        g = np.random.random((batsize, ldim))
        b = EuclideanDistance()
        pred = b.predict(l, r, g)
        print pred.shape
        self.assertEqual(pred.shape, (batsize, seqlen))

    def test_seq_gated_values(self):
        batsize = 10
        ldim = 5
        rdim = 5
        seqlen = 6
        l = np.ones((batsize, ldim))
        r = np.zeros((batsize, seqlen, rdim))
        g = np.random.random((batsize, ldim))
        b = EuclideanDistance()
        pred = b.predict(l, r, g)
        print pred
        print np.sqrt(np.sum(g, axis=-1))
        print pred.shape
        self.assertEqual(pred.shape, (batsize, seqlen))
        self.assertTrue(np.allclose(np.sqrt(np.sum(g, axis=-1)), pred[:, 0]))

    def test_gated(self):
        batsize = 10
        dim = 5
        l = np.random.random((batsize, dim))
        r = np.random.random((batsize, dim))
        g = np.random.random((batsize, dim))
        b = EuclideanDistance()
        pred = b.predict(l, r, g)
        print pred
        b2 = EuclideanDistance()
        prednogate = b2.predict(l, r)
        print prednogate
        self.assertEqual(pred.shape, (batsize,))


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
        pred = b.predict(l, r)
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

    def test_shape_seq(self):
        batsize = 10
        ldim = 5
        rdim = 5
        seqlen = 6
        l = np.random.random((batsize, ldim))
        r = np.random.random((batsize, seqlen, rdim))
        b = DotDistance()
        pred = b.predict(l, r)
        print pred.shape


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
        params = {d.leftblock.W, d.leftblock.b, d.rightblock.W, d.rightblock.b, d.agg}
        self.assertEqual(params, d.get_params())

    def test_seq(self):
        batsize = 6
        ldim = 3
        rdim = 2
        seqlen = 4
        np.random.seed(544)
        l = np.random.random((batsize, ldim))
        r = np.random.random((batsize, seqlen, rdim))
        b = LinearDistance(ldim, rdim, 5)
        pred = b.predict(l, r)
        for k in range(batsize):
            le = l[k]
            re = r[k]
            for i in range(seqlen):
                x = np.dot(le, b.leftblock.W.value.get_value()) + b.leftblock.b.value.get_value()
                y = np.dot(re[i], b.rightblock.W.value.get_value()) + b.rightblock.b.value.get_value()
                z = np.dot(x + y, b.agg.value.get_value())
                self.assertTrue(np.allclose(z, pred[k][i]))


class TestForwardDistance(TestLinearDistance):
    def test_shape(self):
        batsize = 10
        ldim = 5
        rdim = 4
        aggdim = 7
        l = np.random.random((batsize, ldim))
        r = np.random.random((batsize, rdim))
        b = ForwardDistance(ldim, rdim, aggdim)
        pred = b.predict(l, r)
        self.assertEqual(pred.shape, (batsize,))
