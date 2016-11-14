from unittest import TestCase
from teafacto.blocks.seq.oldseqproc import SimpleSeqTransDec
from teafacto.use.recsearch import GreedySearch
import numpy as np


class TestSeqTransDecSearcher(TestCase):

    def test_stop_symbol(self):
        m = SimpleSeqTransDec(indim=20, outdim=10, inpembdim=8, outembdim=9, innerdim=11)
        inpseq = np.random.randint(0, 20, (5, 7)).astype("int32")

        searcher = GreedySearch(m, startsymbol=0, stopsymbol=0)
        searcher.init(5)
        out = searcher.search(inpseq)
        self.assertTrue(np.all(out[0] == np.zeros_like(out[0])))


