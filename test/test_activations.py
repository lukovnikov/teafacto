from unittest import TestCase
import numpy as np
from teafacto.blocks.activations import Softmax, GumbelSoftmax
from teafacto.core import Val


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
            gsmpred = gsmpredf(smpred)
            samples.append(gsmpred[0, :])
        samples = np.concatenate([sample[:, np.newaxis] for sample in samples],
                                axis=1)
        sampleavg = np.average(samples, axis=1)
        print sampleavg
        np.set_printoptions(precision=5, suppress=True)
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
        gsmpred = gsm.predict(smpred, m)

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
        gsm = GumbelSoftmax(temperature=0.3)
        smpred = sm.predict(d)
        gsmpred = gsm.predict(smpred, m)

        np.set_printoptions(precision=5, suppress=True)
        print gsmpred
        predsums = np.sum(gsmpred, axis=-1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))
        self.assertEqual(d.shape, gsmpred.shape)
        self.assertTrue(np.allclose(np.zeros_like(gsmpred[:, :, 2]), gsmpred[:, :, 2]))



