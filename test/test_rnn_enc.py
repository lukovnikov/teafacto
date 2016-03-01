from unittest import TestCase
from teafacto.core.base import *
from teafacto.core.rnn import *
from teafacto.core.rnu import *
from teafacto.core.stack import stack
from teafacto.core.base import tensorops as T
import numpy as np


class SimpleRNNEncoderTest(TestCase):
    expectedparams = ["um", "wm", "uhf", "whf", "u", "w", "bm", "bhf", "b"]
    expectednumberparams = 1

    def setUp(self):
        dim = 50
        self.outdim = 100
        batsize = 1000
        seqlen = 19
        self.enc = RNNEncoder() + GRU(dim=dim, innerdim=self.outdim)
        self.enc = self.doswitches(self.enc)
        self.data = np.random.random((batsize, seqlen, dim))
        self.out = self.enc.predict(self.data)

    def test_output_shape(self):
        self.assertEqual(self.out.shape, (self.data.shape[0], self.outdim))

    def test_all_output_parameters(self):
        outputs = self.enc.wrapply(*self.enc.inputs)
        if issequence(outputs) and len(outputs) > 1:
            outputparamsets = [x.allparams for x in outputs]
            for i in range(len(outputparamsets)):
                for j in range(i, len(outputparamsets)):
                    self.assertSetEqual(outputparamsets[i], outputparamsets[j])
        if issequence(outputs):
            outputs = outputs[0]
        outputparamcounts = {}
        for paramname in [x.name for x in outputs.allparams]:
            if paramname not in outputparamcounts:
                outputparamcounts[paramname] = 0
            outputparamcounts[paramname] += 1
        for (_, y) in outputparamcounts.items():
            self.assertEqual(y, self.expectednumberparams)
        self.assertSetEqual(set(outputparamcounts.keys()), set(self.expectedparams))

    def doswitches(self, enc):
        return enc


class SimpleRNNEncoderTestAllStates(SimpleRNNEncoderTest):
    def doswitches(self, enc):
        return enc.all_states


class SimpleRNNEncoderTestWithOutputs(SimpleRNNEncoderTest):
    def doswitches(self, enc):
        return enc.with_outputs


class SimpleRNNEncoderTestAllStatesWithOutputs(SimpleRNNEncoderTest):
    def doswitches(self, enc):
        return enc.all_states.with_outputs


class StackRNNEncoderTest(SimpleRNNEncoderTest):
    expectednumberparams = 3

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
        self.enc = self.doswitches(self.enc)
        self.data = np.random.random((batsize, seqlen, indim))
        self.out = self.enc.predict(self.data)

    def doswitches(self, enc):
        return enc


class StackRNNEncoderTestAllStates(StackRNNEncoderTest):
    def doswitches(self, enc):
        return enc.all_states


class StackRNNEncoderTestWithOutputs(StackRNNEncoderTest):
    def doswitches(self, enc):
        return enc.with_outputs


class StackRNNEncoderTestAllStatesWithOutputs(StackRNNEncoderTest):
    def doswitches(self, enc):
        return enc.all_states.with_outputs


