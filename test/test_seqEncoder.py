from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.rnn import SeqEncoder, RNNSeqEncoder
from teafacto.blocks.seq.rnu import GRU, LSTM
from teafacto.blocks.basic import IdxToOneHot, VectorEmbed
from teafacto.core.base import Val


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

    def test_output_shape_LSTM(self):
        batsize = 100
        seqlen = 5
        dim = 50
        indim = 13
        m = SeqEncoder(IdxToOneHot(13), LSTM(dim=indim, innerdim=dim))
        data = np.random.randint(0, indim, (batsize, seqlen)).astype("int32")
        mpred = m.predict(data)
        self.assertEqual(mpred.shape, (batsize, dim))

    def test_output_shape_LSTM_2layer(self):
        batsize = 100
        seqlen = 5
        dim = 50
        indim = 13
        dim2 = 40
        m = SeqEncoder(IdxToOneHot(13),
                       LSTM(dim=indim, innerdim=dim),
                       LSTM(dim=dim, innerdim=dim2))
        data = np.random.randint(0, indim, (batsize, seqlen)).astype("int32")
        mpred = m.predict(data)
        self.assertEqual(mpred.shape, (batsize, dim2))

    def test_output_shape_w_mask(self):
        batsize = 2
        seqlen = 5
        dim = 3
        indim = 7
        m = SeqEncoder(IdxToOneHot(indim, maskid=-1), GRU(dim=indim, innerdim=dim)).all_outputs()
        data = np.random.randint(0, indim, (batsize, seqlen)).astype("int32")
        data[:, 0:2] = -1
        weights = np.ones_like(data).astype("float32")
        mpred = m.predict(data, weights)
        self.assertEqual(mpred.shape, (batsize, seqlen, dim))

    def test_mask_dynamic_pad(self):
        batsize = 10
        seqlen = 5
        dim = 6
        indim = 5
        m = SeqEncoder(IdxToOneHot(indim, maskid=-1), GRU(dim=indim, innerdim=dim)).all_outputs()
        data = np.random.randint(0, indim, (batsize, seqlen)).astype("int32")
        rmasker = np.random.randint(2, seqlen, (batsize, )).astype("int32")
        print rmasker
        for i in range(data.shape[0]):
            data[i, rmasker[i]:] = -1
        print data
        pred = m.predict(data)
        np.set_printoptions(precision=3)
        print pred

    def test_mask_no_state_updates(self):
        batsize = 10
        seqlen = 3
        dim = 7
        indim = 5
        m = SeqEncoder(IdxToOneHot(indim, maskid=0), GRU(dim=indim, innerdim=dim))\
            .with_outputs().with_states()
        data = np.random.randint(1, indim, (batsize, seqlen)).astype("int32")
        data[:, 1] = 1
        ndata = np.ones_like(data) * 0
        data = np.concatenate([data, ndata], axis=1)
        data = Val(data)
        finalvar, predvar, statevars = m(data)
        final = finalvar.eval()
        pred = predvar.eval()
        predmask = predvar.mask.eval()
        states = [statevar.eval() for statevar in statevars]
        statemasks = [statevar.mask.eval() if statevar.mask is not None else None
                      for statevar in statevars]
        print predmask
        print statemasks
        print pred.shape
        print states[0].shape, len(states)
        state = states[0]
        print pred[0, :, :3]
        print final[0, :3]
        print state[0, :, :3]
        for i in range(1, state.shape[1]):
            print np.linalg.norm(state[:, i - 1, :] - state[:, i, :])
            if i < seqlen:
                self.assertTrue(not np.allclose(state[:, i - 1, :], state[:, i, :]))
            else:
                self.assertTrue(np.allclose(state[:, i - 1, :], state[:, i, :]))
        for i in range(1, pred.shape[1]):
            print np.linalg.norm(pred[:, i - 1, :] - pred[:, i, :])
            if i < seqlen:
                self.assertTrue(not np.allclose(pred[:, i - 1, :], pred[:, i, :]))
            else:
                self.assertTrue(np.allclose(np.zeros_like(pred[:, i, :]), pred[:, i, :]))

    def test_mask_no_state_updates_multi_layer(self):
        batsize = 10
        seqlen = 3
        dim = 7
        indim = 5
        m = SeqEncoder(IdxToOneHot(indim, maskid=0),
                       GRU(dim=indim, innerdim=dim),
                       GRU(dim=dim, innerdim=dim+1))\
            .all_outputs().with_states()
        data = np.random.randint(1, indim, (batsize, seqlen)).astype("int32")
        data[:, 1] = 1
        ndata = np.ones_like(data) * 0
        data = np.concatenate([data, ndata], axis=1)
        pred, states = m.predict(data)
        state = states[0]
        print "prediction:"
        print pred.shape
        print pred[0, :, :3]
        print "states:"
        for state in states:
            print state.shape
            print state[0, :, :3]
            for i in range(1, state.shape[1]):
                #print np.linalg.norm(state[:, i - 1, :] - state[:, i, :])
                if i < seqlen:
                    self.assertTrue(not np.allclose(state[:, i - 1, :], state[:, i, :]))
                else:
                    self.assertTrue(np.allclose(state[:, i - 1, :], state[:, i, :]))
        for i in range(1, pred.shape[1]):
            #print np.linalg.norm(pred[:, i - 1, :] - pred[:, i, :])
            if i < seqlen:
                self.assertTrue(not np.allclose(pred[:, i - 1, :], pred[:, i, :]))
            else:
                self.assertTrue(np.allclose(np.zeros_like(pred[:, i, :]), pred[:, i, :]))

    def test_mask_no_state_updates_multi_layer_bidir(self):
        batsize = 10
        seqlen = 4
        dim = 7
        indim = 5
        emb = IdxToOneHot(indim, maskid=0)
        m = SeqEncoder.fluent().setembedder(emb)\
            .addlayers(dim=dim, bidir=True)\
            .addlayers(dim=dim+1, bidir=True)\
            .make().with_outputs().with_states()
        data = np.random.randint(1, indim, (batsize, seqlen)).astype("int32")
        data[:, 1] = 1
        ndata = np.ones((batsize, seqlen-2), dtype="int32") * 0
        data = np.concatenate([data, ndata], axis=1)
        finals, pred, states = m.predict(data)

        print "prediction:"
        print pred.shape
        print np.concatenate([finals[0, :2], finals[0, -2:]], axis=-1)
        print np.concatenate([pred[0, :, :2], pred[0, :, -2:]], axis=-1)
        self.assertTrue(np.allclose(np.concatenate([finals[0, :2], finals[0, -2:]], axis=-1),
                                    np.concatenate([pred[0, seqlen-1, :2], pred[0, 0, -2:]], axis=-1)))
        print "states:"
        for state in states[::-1]:
            print state.shape
            print np.concatenate([state[0, :, :2], state[0, :, -2:]], axis=-1)
            for i in range(1, state.shape[1]):
                #print np.linalg.norm(state[:, i - 1, :] - state[:, i, :])
                if i < seqlen:
                    self.assertTrue(not np.allclose(state[:, i - 1, :state.shape[2]/2], state[:, i, :state.shape[2]/2]))
                else:
                    self.assertTrue(np.allclose(state[:, i - 1, :state.shape[2]/2], state[:, i, :state.shape[2]/2]))
        for i in range(1, pred.shape[1]):
            #print np.linalg.norm(pred[:, i - 1, :] - pred[:, i, :])
            if i < seqlen:
                self.assertTrue(not np.allclose(pred[:, i - 1, :], pred[:, i, :]))
            else:
                self.assertTrue(np.allclose(np.zeros_like(pred[:, i, :]), pred[:, i, :]))

    def test_mask_propagation_all_states(self):
        m = SeqEncoder(VectorEmbed(maskid=0, indim=100, dim=7),
                       GRU(dim=7, innerdim=30)).all_outputs()
        data = np.random.randint(1, 100, (5, 3), dtype="int32")
        ndata = np.zeros_like(data)
        data = np.concatenate([data, ndata], axis=1)

        dataval = Val(data)
        embvar = m.embedder(dataval)
        embpred = embvar.eval()
        embmaskpred = embvar.mask.eval()

        encvar = m(dataval)
        encpred = encvar.eval()
        encmaskpred = encvar.mask.eval()
        print encpred.shape
        print encmaskpred.shape
        print encmaskpred
        self.assertTrue(np.allclose(encmaskpred, embmaskpred))


class TestRNNSeqEncoder(TestCase):
    def test_bidir(self):
        np.set_printoptions(precision=3, suppress=True)
        m = RNNSeqEncoder(indim=20, inpembdim=5, innerdim=(10, 10), bidir=True, maskid=0).with_outputs()
        xval = np.random.randint(1, 20, (7, 3))
        xval = np.concatenate([xval, np.zeros_like(xval)], axis=1)
        x = Val(xval)
        fmp, mp = m(x)
        fmpval, mpval = fmp.eval(), mp.eval()
        print "finals"
        print fmpval[:, [0,1,2,-3,-2,-1]]
        print "outs"
        print mpval[:, :, [0,1,2,-3,-2,-1]]
        self.assertTrue(np.allclose(fmpval[:, :10], mpval[:, 2, :10]))
        self.assertTrue(np.allclose(fmpval[:, 10:], mpval[:, 0, 10:]))
        mpm = mp.mask
        self.assertEqual(np.sum(mpm.eval() - xval > 0), 0)
