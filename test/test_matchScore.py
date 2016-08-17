from unittest import TestCase
from teafacto.blocks.match import MatchScore, SeqMatchScore, CosineDistance
from teafacto.blocks.seqproc import SeqUnroll
from teafacto.blocks.basic import VectorEmbed
import numpy as np

class TestMatchScore(TestCase):

    def test_seq_scoring(self):
        vocsize = 100
        dim = 10
        numsam = 17
        seqlen = 5
        ve = VectorEmbed(vocsize, dim)
        m = SeqMatchScore(SeqUnroll(ve), SeqUnroll(ve), scorer=CosineDistance())

        data = np.random.randint(0, vocsize, (numsam, seqlen))
        #print data.shape
        pred = m.predict(data, data)
        #print pred
        self.assertTrue(np.allclose(np.ones_like(pred)*seqlen*1., pred))

