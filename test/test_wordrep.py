from unittest import TestCase
from teafacto.blocks.word.wordrep import *
from teafacto.blocks.basic import VectorEmbed
from teafacto.blocks.seq import RNNSeqEncoder
from teafacto.core import Val
import numpy as np


class TestWordEmbCharEncConcat(TestCase):
    def setUp(self):
        self.vocsize = 500
        self.embdim = 50
        self.numchars = 200
        self.charembdim = 20
        self.wordemb = VectorEmbed(self.vocsize, self.embdim, maskid=0)
        self.charenc = RNNSeqEncoder.fluent()\
            .vectorembedder(self.numchars, self.charembdim)\
            .addlayers(self.embdim, bidir=True)\
            .addlayers(self.embdim, bidir=False).make()

        self.batsize, self.seqlen, self.charseqlen = 15, 7, 9

        self.data = self._gen_data(self.batsize, self.seqlen, self.charseqlen,
                                   self.vocsize, self.numchars)

    def _gen_data(self, batsize, seqlen, charseqlen, numwords, numchars):
        worddata = np.random.randint(0, numwords, (batsize, seqlen, 1))
        chardata = np.random.randint(0, numchars, (batsize, seqlen, charseqlen))
        data = np.concatenate([worddata, chardata], axis=-1)
        return data

    def test_shapes_and_wordembs(self):
        b = WordEmbCharEncConcat(self.wordemb, self.charenc)
        x = Val(self.data) + 0
        y = b(x)
        pred = y.eval()
        print pred.shape
        self.assertEqual((self.batsize, self.seqlen, self.embdim * 2),
                         pred.shape)

        wemb = self.wordemb(x[:, :, 0])
        wemb = wemb.eval()
        print wemb.shape
        self.assertTrue(np.allclose(wemb, pred[:, :, :50]))


class TestWordEmbCharEncGate(TestWordEmbCharEncConcat):
    def test_shapes_and_wordembs(self):
        pass

    def test_shapes(self):
        b = WordEmbCharEncGate(self.wordemb, self.charenc)
        x = Val(self.data) + 0
        y = b(x)
        pred = y.eval()
        print pred.shape
        self.assertEqual(pred.shape, (self.batsize, self.seqlen, self.embdim))


class TestWordEmbCharEncCtxGate(TestWordEmbCharEncGate):
    def setUp(self):
        super(TestWordEmbCharEncCtxGate, self).setUp()
        self.gate = RNNSeqEncoder.fluent()\
            .noembedder(self.embdim * 2)\
            .addlayers(self.embdim, bidir=True)\
            .add_forward_layers(self.embdim, activation=Sigmoid)\
            .make()\
            .all_outputs()

    def test_shapes(self):
        b = WordEmbCharEncCtxGate(self.wordemb, self.charenc, gate_enc=self.gate)
        x = Val(self.data) + 0
        y = b(x)
        pred = y.eval()
        print pred.shape
        self.assertEqual(pred.shape, (self.batsize, self.seqlen, self.embdim))
