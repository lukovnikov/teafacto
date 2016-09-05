from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.rnn import SeqEncoder, MaskSetMode
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.basic import IdxToOneHot


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


    def test_mask_dynamic_pad(self):
        batsize = 10
        seqlen = 5
        dim = 6
        indim = 5
        m = SeqEncoder(IdxToOneHot(indim), GRU(dim=indim, innerdim=dim)).maskoption(-1).all_outputs
        data = np.random.randint(0, indim, (batsize, seqlen)).astype("int32")
        rmasker = np.random.randint(2, seqlen, (batsize, )).astype("int32")
        print rmasker
        for i in range(data.shape[0]):
            data[i, rmasker[i]:] = -1
        print data
        pred = m.predict(data)
        print pred

    def test_mask_no_state_updates(self):
        batsize = 10
        seqlen = 3
        dim = 7
        indim = 5
        m = SeqEncoder(IdxToOneHot(indim), GRU(dim=indim, innerdim=dim)).maskoption(-1).all_outputs
        data = np.random.randint(0, indim, (batsize, seqlen)).astype("int32")
        data[:, 1] = 0
        ndata = np.ones_like(data) * -1
        data = np.concatenate([data, ndata], axis=1)
        pred = m.predict(data)
        for i in range(1, pred.shape[1]):
            print np.linalg.norm(pred[:, i - 1, :] - pred[:, i, :])
            if i < seqlen:
                self.assertTrue(not np.allclose(pred[:, i - 1, :], pred[:, i, :]))
            else:
                self.assertTrue(np.allclose(pred[:, i - 1, :], pred[:, i, :]))

    def test_mask_zero_mask_with_custom_maskid(self):
        batsize = 10
        seqlen = 3
        dim = 7
        indim = 5
        m = SeqEncoder(IdxToOneHot(indim), GRU(dim=indim, innerdim=dim)).maskoptions(-1, MaskSetMode.ZERO).all_outputs
        data = np.random.randint(0, indim, (batsize, seqlen)).astype("int32")
        data[:, 1] = 0
        ndata = np.ones_like(data) * -1
        data = np.concatenate([data, ndata], axis=1)
        pred = m.predict(data)
        for i in range(pred.shape[1]):
            print np.linalg.norm(pred[:, i - 1, :] - pred[:, i, :])
            if i < seqlen:
                for j in range(pred.shape[0]):
                    self.assertTrue(np.linalg.norm(pred[j, i, :]) > 0.0)
            else:
                for j in range(pred.shape[0]):
                    self.assertTrue(np.linalg.norm(pred[j, i, :]) == 0.0)
