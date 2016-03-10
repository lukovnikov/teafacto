from unittest import TestCase

from teafacto.blocks.rnn import SeqDecoder
from teafacto.blocks.rnu import GRU, LSTM
from teafacto.blocks.basic import MatDot, Softmax, IdxToOneHot
from teafacto.core.base import tensorops as T
from teafacto.core.stack import stack
from teafacto.core.base import param, asblock

import numpy as np


class TestSimpleRNNDecoder(TestCase):
    def setUp(self):
        self.batsize = 70
        self.vocsize = 27
        self.encdim = 44
        self.hdim = 33
        self.statedim = 50
        self.seqlen = 30
        self.encodings_data = [np.random.random((self.batsize, self.encdim))]
        self.stackdata = np.random.random((self.batsize, self.seqlen, self.vocsize+self.encdim))
        self.stk = stack(
            GRU(dim=self.vocsize+self.encdim, innerdim=self.hdim),
            LSTM(dim=self.hdim, innerdim=self.statedim),
            MatDot(indim=self.statedim, dim=self.vocsize),
            Softmax())
        self.dec = SeqDecoder(IdxToOneHot(self.vocsize), self.stk, indim=self.vocsize, seqlen=self.seqlen)

    def test_rnndecoder_output_shape(self):
        outvals = self.dec.predict(*self.encodings_data)
        self.assertEqual(outvals.shape, (self.batsize, self.seqlen, self.vocsize))

    def test_decoder_stack_output_shape(self):
        stkoutval = self.stk.predict(self.stackdata)
        self.assertEqual(stkoutval.shape, (self.batsize, self.seqlen, self.vocsize))

    def test_decoder_stack_prediction_shape_after_rnndecoder_prediction(self):
        outvals = self.dec.predict(*self.encodings_data)
        self.assertEqual(outvals.shape, (self.batsize, self.seqlen, self.vocsize))
        stkoutval = self.stk.predict(self.stackdata)
        self.assertEqual(stkoutval.shape, (self.batsize, self.seqlen, self.vocsize))


