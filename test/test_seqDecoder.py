from unittest import TestCase
from teafacto.blocks.rnn import SeqDecoder
from teafacto.blocks.rnu import GRU
from teafacto.blocks.basic import IdxToOneHot
import numpy as np


class TestSeqDecoder(TestCase):
    def test_vector_out(self):
        decdim = 50
        outvocsize = 17
        outemb = IdxToOneHot(outvocsize)
        outembdim = outvocsize
        decrnus = [GRU(dim=outvocsize, innerdim=decdim)]
        dec = SeqDecoder([outemb]+decrnus, innerdim=decdim)

        ctxdata = np.random.random((1, decdim))
        seqdata = np.asarray([[2, 3, 4]])
        pred = dec.predict(ctxdata, seqdata)
        self.assertEqual(pred.shape, (1, 3, outvocsize))

        dec = SeqDecoder([outemb]+decrnus, innerdim=decdim, softmaxoutblock=False)
        pred = dec.predict(ctxdata, seqdata)
        self.assertEqual(pred.shape, (1, 3, decdim))
