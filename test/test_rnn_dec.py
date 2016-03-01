from unittest import TestCase
import numpy as np
from teafacto.blocks.rnn import RNNDecoder
from teafacto.blocks.rnu import GRU, LSTM
from teafacto.blocks.core import *
from teafacto.blocks.stack import *
from teafacto.blocks.core import tensorops as T


class TestSimpleRNNDecoder(TestCase):
    def setUp(self):
        self.batsize = 70
        self.vocsize = 27
        self.hdim = 33
        self.statedim = 50
        self.seqlen = 30
        self.data = [np.random.random((self.batsize, self.statedim)), np.random.random((self.batsize, self.hdim))]
        self.stackdata = np.random.random((self.batsize, self.seqlen, self.vocsize))
        self.O = param((self.statedim, self.vocsize), name="OOOO").uniform()
        Obi = asblock(lambda x: T.nnet.softmax(T.dot(x, self.O)))
        self.stk = stack(GRU(dim=self.vocsize, innerdim=self.hdim), LSTM(dim=self.hdim, innerdim=self.statedim), Obi)
        self.dec = RNNDecoder(self.stk, dim=self.vocsize, hlimit=self.seqlen)

    def test_rnndecoder_output_shape(self):
        outvals = self.dec.predict(*self.data)
        self.assertEqual(outvals.shape, (self.batsize, self.seqlen, self.vocsize))

    def test_decoder_stack_output_shape(self):
        stkoutval = self.stk.predict(self.stackdata)
        self.assertEqual(stkoutval.shape, (self.batsize, self.seqlen, self.vocsize))

    def test_decoder_stack_prediction_shape_after_rnndecoder_prediction(self):
        outvals = self.dec.predict(*self.data)
        self.assertEqual(outvals.shape, (self.batsize, self.seqlen, self.vocsize))
        stkoutval = self.stk.predict(self.stackdata)
        self.assertEqual(stkoutval.shape, (self.batsize, self.seqlen, self.vocsize))


