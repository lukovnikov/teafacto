from unittest import TestCase
from teafacto.blocks.seq.rnu import PPGRU
import numpy as np


class TestPPGRU(TestCase):
    def test_shapes(self):
        indim = 10
        innerdim = 15
        batsize = 5
        seqlen = 7
        rnu = PPGRU(dim=indim, innerdim=innerdim, push_gates_extra_out=False)
        data = np.random.random((batsize, seqlen, indim)).astype("float32")

        pred = rnu.predict(data)

        print pred.shape
        self.assertEqual(pred.shape, (batsize, seqlen, innerdim))

    def test_shapes_with_pg(self):
        indim = 10
        innerdim = 15
        batsize = 5
        seqlen = 7
        rnu = PPGRU(dim=indim, innerdim=innerdim, push_gates_extra_out=True)
        data = np.random.random((batsize, seqlen, indim)).astype("float32")

        pred, extra = rnu.predict(data, _extra_outs=["pushgates"])
        print pred.shape
        print extra["pushgates"].shape
        self.assertEqual(pred.shape, (batsize, seqlen, innerdim))
        self.assertEqual(extra["pushgates"].shape, (seqlen, batsize, 2, innerdim))
