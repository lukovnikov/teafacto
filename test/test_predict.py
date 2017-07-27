from unittest import TestCase

from teafacto.examples.dummy import Dummy
from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt
from teafacto.plot.attention import AttentionPlotter
import numpy as np


class TestBlockPredictor(TestCase):
    def test_dummy_pred(self):
        m = Dummy(20, 5)
        data = np.random.randint(0, 20, (50,))
        pred = m.predict(data)
        print pred
