from unittest import TestCase
from teafacto.blocks.rnn import SeqEncoder
from teafacto.blocks.rnu import GRU
from teafacto.blocks.basic import IdxToOneHot, MatDot, Softmax
import numpy as np

class TestSeqEncoder(TestCase):
    def test_output_shape(self):
        batsize = 100
        seqlen = 5
        dim = 50
        indim = 13
        m = SeqEncoder(IdxToOneHot(13), GRU(dim=indim, innerdim=dim))
        data = np.random.randint(0, indim, (batsize, seqlen)).astype("int32")
        mpred = m.predict(data)
        self.assertEqual(mpred.shape, (batsize, dim))

    def test_output_shape_w_mask(self):
        batsize = 2
        seqlen = 5
        dim = 3
        indim = 7
        m = SeqEncoder(IdxToOneHot(indim), GRU(dim=indim, innerdim=dim)).all_outputs
        data = np.random.randint(0, indim, (batsize, seqlen)).astype("int32")
        mask = np.zeros_like(data).astype("float32")
        mask[:, 0:2] = 1
        weights = np.ones_like(data).astype("float32")
        mpred = m.predict(data, weights, mask)
        self.assertEqual(mpred.shape, (batsize, seqlen, dim))
