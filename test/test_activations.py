from unittest import TestCase
import numpy as np
from teafacto.blocks.activations import *
from teafacto.core import Val, param


class TestSoftmax(TestCase):
    def test_softmax_normal(self):
        b = Softmax()
        d = np.random.random((5, 3))
        pred = b.predict(d)
        predsums = np.sum(pred, axis=1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))
        self.assertEqual(d.shape, pred.shape)

    def test_softmax_3D(self):
        b = Softmax()
        d = np.random.random((5, 4, 3))
        pred = b.predict(d)
        predsums = np.sum(pred, axis=2)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))
        self.assertEqual(d.shape, pred.shape)

    def test_softmax_5D(self):
        b = Softmax()
        d = np.random.random((7, 6, 5, 4, 3))
        pred = b.predict(d)
        predsums = np.sum(pred, axis=-1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))
        self.assertEqual(d.shape, pred.shape)

    def test_softmax_normal_masked(self):
        b = Softmax()
        d = np.random.random((5, 3))
        m = np.ones_like(d)
        m[:, 2] = 0
        pred = b.predict(d, m)
        print pred
        self.assertTrue(np.allclose(np.zeros_like(pred[:, 2]), pred[:, 2]))
        self.assertEqual(d.shape, pred.shape)
        predsums = np.sum(pred, axis=-1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))

    def test_softmax_normal_masked_maxhot(self):
        b = Softmax(maxhot=True)
        d = np.random.random((5, 3))
        m = np.ones_like(d)
        m[:, 2] = 0
        pred = b.predict(d, m)
        print pred
        self.assertTrue(np.allclose(np.zeros_like(pred[:, 2]), pred[:, 2]))
        self.assertEqual(d.shape, pred.shape)
        predsums = np.sum(pred, axis=-1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))

    def test_softmax_3D_masked(self):
        b = Softmax()
        d = np.random.random((5, 4, 3))
        m = np.ones_like(d)
        m[:, :, 2] = 0
        pred = b.predict(d, m)
        self.assertTrue(np.allclose(np.zeros_like(pred[:, :, 2]), pred[:, :, 2]))
        self.assertEqual(d.shape, pred.shape)
        predsums = np.sum(pred, axis=-1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))

    def test_softmax_3D_masked_maxhot(self):
        b = Softmax(maxhot=True)
        d = np.random.random((5, 4, 3))
        m = np.ones_like(d)
        m[:, :, 2] = 0
        pred = b.predict(d, m)
        print pred
        self.assertTrue(np.allclose(np.zeros_like(pred[:, :, 2]), pred[:, :, 2]))
        self.assertEqual(d.shape, pred.shape)
        predsums = np.sum(pred, axis=-1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))

    def test_softmax_3D_prop_seq_mask(self):
        b = Softmax()
        d = np.random.random((5, 4, 3))
        m = np.ones((5, 4))
        m[:, 2:] = 0
        d = Val(d) + 0
        m = Val(m) + 0
        d.mask = m
        pred = b(d)
        predmask = pred.mask.eval()
        pred = pred.eval()
        self.assertTrue(np.allclose(predmask, m.eval()))
        predsums = np.sum(pred, axis=-1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))

    def test_softmax_normal_with_temperature(self):
        b = Softmax(temperature=1e-6)
        d = np.random.random((5, 3))
        pred = b.predict(d)
        print pred


class TestGumbelSoftmax(TestCase):
    def test_gumbel_softmax(self):
        d = np.random.random((5, 3))
        d[0, 0] = 2.
        sm = Softmax()
        gsm = GumbelSoftmax(temperature=1e-10, _alwaysrandom=True)
        smpred = sm.predict(d)
        samples = []
        gsmpredf = gsm.predict
        for i in range(100000):
            gsmpred = gsmpredf(d)
            samples.append(gsmpred[0, :])
        samples = np.concatenate([sample[:, np.newaxis] for sample in samples],
                                axis=1)
        sampleavg = np.average(samples, axis=1)
        print sampleavg
        np.set_printoptions(precision=5, suppress=True)
        print d[0]
        print smpred
        print gsmpred
        predsums = np.sum(gsmpred, axis=-1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))
        self.assertEqual(d.shape, gsmpred.shape)
        self.assertTrue(np.allclose(smpred[0, :], sampleavg, rtol=1e-1))

    def test_masked_gumbel_softmax(self):
        d = np.random.random((5, 3))
        m = np.ones_like(d)
        m[:, 2] = 0
        sm = Softmax()
        gsm = GumbelSoftmax(temperature=0.3)
        smpred = sm.predict(d)
        gsmpred = gsm.predict(d, m)

        np.set_printoptions(precision=5, suppress=True)
        print gsmpred
        predsums = np.sum(gsmpred, axis=-1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))
        self.assertEqual(d.shape, gsmpred.shape)
        self.assertTrue(np.allclose(np.zeros_like(gsmpred[:, 2]), gsmpred[:, 2]))

    def test_masked_3D_gumbel_softmax(self):
        d = np.random.random((5, 4, 3))
        m = np.ones_like(d)
        m[:, :, 2] = 0
        sm = Softmax()
        gsm = GumbelSoftmax(temperature=0.3, _alwaysrandom=True)
        smpred = sm.predict(d)
        gsmpred = gsm.predict(smpred, m)

        np.set_printoptions(precision=5, suppress=True)
        print gsmpred
        predsums = np.sum(gsmpred, axis=-1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))
        self.assertEqual(d.shape, gsmpred.shape)
        self.assertTrue(np.allclose(np.zeros_like(gsmpred[:, :, 2]), gsmpred[:, :, 2]))

    def test_maxhot(self):
        d = np.random.random((5,4))
        sm = GumbelSoftmax(maxhot=True)
        pred = sm.predict(d)

        print pred


class TestMaxHot(TestCase):
    def test_forward(self):
        x = Val(np.random.random((10, 5)))
        y = MaxHot(ste=True)(x)
        xval = x.d.eval()
        xmaxhot = (xval == np.max(xval, axis=-1, keepdims=True))*1.
        self.assertTrue(np.allclose(xmaxhot, y.eval()))

    def test_train(self):
        p = param((5,), name="testparam").uniform()
        maxhot = MaxHot(ste=True)
        m = asblock(lambda x: maxhot(x + p) + 1e-6)
        data = np.random.random((100, 5)).astype("float32")
        gold = np.ones((100,)).astype("int32") * 4
        res = m(Val(data))
        print res.eval()
        m.train([data], gold).bitspersym().adadelta(lr=1).train(10, 100)
        np.set_printoptions(suppress=True, precision=2)
        print p.d.eval()
        print np.max(data[:, :4]) - np.max(data[:, 4])
        self.assertTrue(p.d.eval()[4] + np.max(data[:, 4]) > np.max(data[:, :4]))


class TestThreshold(TestCase):
    def test_forward(self):
        x = Val(np.random.random((10, 5)))
        y = Threshold(0.5, ste=True)(x)
        xval = x.d.eval()
        xt = (xval > 0.5) * 1.
        self.assertTrue(np.allclose(xt, y.eval()))

    def test_forward_ste_sigm(self):
        x = Val(np.random.random((10, 5))*2-1)
        y = Threshold(0.5, ste="sigmoid")(x)
        xval = x.d.eval()
        xt = (xval > 0) * 1.
        self.assertTrue(np.allclose(xt, y.eval()))

    def test_forward_ste_hardsigm(self):
        x = Val(np.random.random((10, 5))*2-1)
        y = Threshold(0.5, ste="hardsigmoid")(x)
        xval = x.d.eval()
        xt = (xval > 0) * 1.
        self.assertTrue(np.allclose(xt, y.eval()))

    def test_train(self):
        p = param((1,), name="testparam").uniform()
        thresh = Threshold(0.5, ste=True)
        m = asblock(lambda x: thresh(x+0.2+T.repeat(p, x.shape[0], axis=0)))
        data = np.random.random((100,)).astype("float32")
        data -= np.min(data)
        gold = np.ones((100,))
        res = m(Val(data))
        print res.eval()
        m.train([data], gold).squared_error().adadelta(lr=1).train(10, 25)
        print p.d.eval()
        self.assertTrue(np.allclose(p.d.eval()[0], 0.3, atol=0.01))


class TestStochasticThreshold(TestCase):
    def test_forward(self):
        x = Val(np.random.random((10, 5))*2-1)
        y = StochasticThreshold(ste=True, detexe=False)(x)
        xval = x.d.eval()
        xt = (xval > 0.5) * 1.
        print y.eval()
        self.assertEqual(y.eval().shape, (10, 5))

    def test_forward_ste_sigmoid(self):
        x = Val(np.random.random((10, 5))*2-1)
        y = StochasticThreshold(ste="sigmoid", detexe=False)(x)
        xval = x.d.eval()
        xt = (xval > 0.5) * 1.
        print y.eval()
        self.assertEqual(y.eval().shape, (10, 5))

    def test_forward_ste_hardsigmoid(self):
        x = Val(np.random.random((10, 5)))
        y = StochasticThreshold(ste="hardsigmoid", detexe=False)(x)
        xval = x.d.eval()
        xt = (xval > 0.5) * 1.
        print y.eval()
        self.assertEqual(y.eval().shape, (10, 5))


