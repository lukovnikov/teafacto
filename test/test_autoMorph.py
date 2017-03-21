from unittest import TestCase
from teafacto.blocks.seq.memnn import AutoMorph
from teafacto.blocks.basic import VectorEmbed
import numpy as np


class TestAutoMorph(TestCase):
    def test_shapes(self):
        charemb = VectorEmbed(256, 20, maskid=0)
        m = AutoMorph(memlen=100, memkeydim=30, memvaldim=31,
                      charemb=charemb, outdim=32)
        batsize = 15
        seqlen = 6
        data = np.random.randint(0, 256, (batsize, seqlen))
        pred = m.predict(data)
        self.assertEqual(pred.shape, (batsize, seqlen, 32))
