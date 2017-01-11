from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.rnn import SeqDecoder
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.basic import IdxToOneHot
from teafacto.core.base import Val


class TestSeqDecoder(TestCase):
    def test_vector_out(self):
        decdim = 50
        outvocsize = 17
        outemb = IdxToOneHot(outvocsize)
        outembdim = outvocsize
        decrnus = [GRU(dim=outvocsize, innerdim=decdim)]
        dec = SeqDecoder([outemb]+decrnus, innerdim=decdim*2, outconcat=True, inconcat=False)

        ctxdata = np.random.random((2, decdim)).astype("float32")
        seqdata = np.asarray([[2, 3, 4], [2, 3, 4]])
        pred = dec.predict(ctxdata, seqdata)
        self.assertEqual(pred.shape, (2, 3, outvocsize))

        dec = SeqDecoder([outemb]+decrnus, innerdim=decdim*2, softmaxoutblock=False, outconcat=True, inconcat=False)
        pred = dec.predict(ctxdata, seqdata)
        self.assertEqual(pred.shape, (2, 3, decdim*2))

