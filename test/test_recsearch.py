from unittest import TestCase
from teafacto.blocks.seq.oldseqproc import SimpleSeqTransDec
from teafacto.use.recsearch import GreedySearch, SeqTransDecSearcher
import numpy as np


class TestSeqTransDecSearcher(TestCase):
    def test_old_search_same_as_new(self):
        m = SimpleSeqTransDec(indim=20, outdim=10, inpembdim=8, outembdim=9, innerdim=11)
        inpseq = np.random.randint(0, 20, (5, 7)).astype("int32")

        oldsearcher = SeqTransDecSearcher(m)
        oldsearcher.init(5)
        oldout = oldsearcher.search(inpseq)
        self.assertEqual(oldout[0].shape, (5, 7))

        newsearcher = GreedySearch(m, startsymbol=0)
        newsearcher.init(5)
        newout = newsearcher.search(inpseq)

        self.assertTrue(np.all(oldout[0] == newout[0]))
        self.assertTrue(np.allclose(oldout[1], newout[1]))

    def test_stop_symbol(self):
        m = SimpleSeqTransDec(indim=20, outdim=10, inpembdim=8, outembdim=9, innerdim=11)
        inpseq = np.random.randint(0, 20, (5, 7)).astype("int32")

        searcher = GreedySearch(m, startsymbol=0, stopsymbol=0)
        searcher.init(5)
        out = searcher.search(inpseq)
        self.assertTrue(np.all(out[0] == np.zeros_like(out[0])))


