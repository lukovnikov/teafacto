from unittest import TestCase
from teafacto.blocks.seqproc import SimpleSeq2Vec
from teafacto.blocks.match import MatchScore
import numpy as np
import theano


class TestSimpleSeq2Vec(TestCase):
    def setUp(self):
        enc = SimpleSeq2Vec(indim=100, inpembdim=10, innerdim=20)
        x = np.random.randint(0, 100, (33, 5))
        o = enc.autobuild(x)
        self.o = o[1][0]
        m = MatchScore(enc, enc)
        mo = m.autobuild(x, x)
        self.mo = mo[1][0]

    def test_output(self):
        print self.mo, self.mo.ndim
        #theano.printing.pydotprint(self.mo.d, "debug.png")
