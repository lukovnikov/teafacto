from unittest import TestCase
import numpy as np

from teafacto.search import SeqTransDecSearch
from teafacto.blocks.rnn import SimpleSeqTransDec
from teafacto.util import ticktock as TT


class TestSeqTransDecSearch(TestCase):
    def test_decode_shape(self):
        tt = TT("timer")
        batsize = 10
        seqlen = 200
        model = SimpleSeqTransDec(indim=200, outdim=50, inpembdim=20, outembdim=20, innerdim=[40, 30])
        searcher = SeqTransDecSearch(model)
        inpval = np.random.randint(0, 50, (batsize, seqlen)).astype("int32")
        tt.tick()
        out, probs = searcher.decode(inpval)
        tt.tock("decoded")
        tt.tick()
        out2, probs2 = searcher.decode2(inpval)
        tt.tock("decoded using previous one")
        self.assertTrue(np.allclose(out, out2))
        self.assertTrue(np.allclose(probs, probs2))
