from unittest import TestCase
from teafacto.core.base import tensorops as T, Val, Var, asblock
import numpy as np


class TestScan(TestCase):
    def test_scan_basic(self):
        def rec(x):
            return x, {x: x + 1}
        ret = T.scan(rec, non_sequences=Val(0), outputs_info=[None], n_steps=5)
        pred = ret.eval()
        self.assertTrue(np.allclose(pred, range(5)))
        print pred

    def test_scan_basic_with_arg(self):
        x = Val(np.random.random((10, 5)))
        def rec(x, acc_tm1):
            acc_t = acc_tm1 + x
            return x ** 2, acc_t
        ret = T.scan(rec, sequences=x, outputs_info=[None, Val(np.zeros((5,)))])
        mappred = ret[0].eval()
        accpred = ret[1].eval()
        print mappred
        print accpred

    def test_scan_basic_with_arg_and_mask(self):
        x = Val(np.random.random((3, 5, 2)))
        m = Val(np.asarray([[1,1,1,0,0], [1,1,1,0,0], [1,1,1,0,0]], dtype="int32"))
        x.mask = m
        def rec(xe, acc_tm1):
            me = xe.mask
            xe = T.cast(xe * me.dimadd(1), "float32")
            acc_t = acc_tm1 + xe
            acc_t.mask = me
            retval = xe ** 2
            retval.mask = me
            return xe ** 2, acc_t
        xr = x.dimswap(0, 1)
        xm = x.mask.dimswap(0, 1)
        xr.mask = xm
        ret = T.scan(rec, sequences=xr, outputs_info=[None, Val(np.zeros((3, 2)))])
        mappred = ret[0].dimswap(0, 1).eval()
        accpred = ret[1].dimswap(0, 1).eval()
        accmaskpred = ret[1].mask.dimswap(0,1).eval()
        self.assertTrue(ret[1].mask is not None)
        print mappred
        print mappred.shape
        print accpred
        print accpred.shape
        print accmaskpred
        print mappred[:, 3:].shape
        self.assertTrue(np.allclose(mappred[:, 3:], np.zeros_like(mappred[:, 3:])))

    def test_scan_nested(self):
        def outerrec(x):
            z = Val(0)
            y = T.scan(innerrec, non_sequences=[x, z], n_steps=3)
            return y, {z: Val(0)}

        def innerrec(x, z):
            return [x, z], {x: x + 1, z: z + 1}

        scanblock = asblock(lambda: T.scan(outerrec, non_sequences=Val(0), n_steps=4))

        pred = scanblock.predict()
        print pred
        self.assertTrue(np.allclose(pred[0], np.arange(12).reshape((4, 3))))
        self.assertTrue(np.allclose(pred[1], np.repeat(np.arange(3)[np.newaxis, :], 4, axis=0)))
