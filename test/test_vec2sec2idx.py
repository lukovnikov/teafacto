from unittest import TestCase
import numpy as np

from teafacto.blocks.attention import vec2seq, seq2idx, idx2seq


class TestVecSeqIdx(TestCase):
    def setUp(self):
        self.encdim = 44
        self.innerdim = 50
        self.seqlen = 5
        self.numchars = 7
        self.numwords = 15
        self.indim = 20
        self.batsize = 100

    def test_vec2seq_shape(self):
        b = vec2seq(encdim=self.encdim, indim=self.indim, seqlen=self.seqlen, innerdim=self.innerdim, vocsize=self.numchars)
        data = np.random.random((self.batsize, self.indim))
        p = b.predict(data)
        self.assertEqual(p.shape, (self.batsize, self.seqlen, self.numchars))

    def test_seq2idx_shape(self):
        b = seq2idx(invocsize=self.numchars, outvocsize=self.numwords, innerdim=self.innerdim)
        data = np.random.randint(0, self.numchars, (self.batsize, self.seqlen))
        p = b.predict(data)
        self.assertEqual(p.shape, (self.batsize, self.numwords))

    def test_idx2seq_shape(self):
        b = idx2seq(encdim=self.encdim, invocsize=self.numwords, outvocsize=self.numchars, seqlen=self.seqlen, innerdim=self.innerdim)
        data = np.random.randint(0, self.numwords, (self.batsize, ))
        p = b.predict(data)
        self.assertEqual(p.shape, (self.batsize, self.seqlen, self.numchars))
