from unittest import TestCase
from teafacto.core.base import tensorops as T, Val, Var, asblock


class TestScan(TestCase):
    def test_scan_basic(self):
        def rec(x):
            return x, {x: x + 1}
        ret = T.scan(rec, non_sequences=Val(0), outputs_info=[None], n_steps=5)
        pred = ret.eval()
        print pred

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
