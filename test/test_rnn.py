from unittest import TestCase
from teafacto.blocks.core import *
from teafacto.blocks.rnn import *
from teafacto.blocks.rnu import *
from teafacto.blocks.stack import stack
from teafacto.blocks.core import tensorops as T
import numpy as np


class SimpleRNNEncoderTest(TestCase):
    def setUp(self):
        dim = 50
        self.innerdim = 100
        batsize = 1000
        seqlen = 19
        self.enc = RNNEncoder() + GRU(dim=dim, innerdim=self.innerdim)
        self.data = np.random.random((batsize, seqlen, dim))
        self.out = self.enc.predict(self.data)

    def test_output_shape(self):
        self.assertEqual(self.out.shape, (self.data.shape[0], self.innerdim))

    def test_parameter_propagation(self):
        # TODO
        pass


class StackRNNEncoderTest(TestCase):
    def setUp(self):
        batsize = 1000
        seqlen = 19
        indim = 71
        hdim = 51
        hdim2 = 61
        self.outdim = 47
        self.enc = RNNEncoder(GRU(dim=indim, innerdim=hdim),
                              GRU(dim=hdim, innerdim=hdim2),
                              GRU(dim=hdim2, innerdim=self.outdim))
        self.data = np.random.random((batsize, seqlen, indim))
        self.out = self.enc.predict(self.data)

    def test_output_shape(self):
        self.assertEqual(self.out.shape, (self.data.shape[0], self.outdim))


