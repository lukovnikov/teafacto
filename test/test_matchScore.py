from unittest import TestCase
from teafacto.blocks.match import MatchScore, SeqMatchScore, CosineDistance
from teafacto.blocks.seqproc import SeqUnroll
from teafacto.blocks.basic import VectorEmbed
import numpy as np

class TestMatchScore(TestCase):

    def test_seq_scoring(self):
        vocsize = 100
        dim = 10
        ve = VectorEmbed(vocsize, dim)
        m = SeqMatchScore(SeqUnroll(ve), SeqUnroll(ve), scorer=CosineDistance())

        data = np.random.randint(0, vocsize, (5, 3))
        pred = m.predict(data, data)
        self.assertTrue(np.allclose(np.ones_like(pred)*3., pred))

