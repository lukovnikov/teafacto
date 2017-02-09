from unittest import TestCase
from teafacto.blocks.seq.rnn import RNNWithoutInput
from teafacto.core.base import asblock
import numpy as np


class TestRNNWithoutInput(TestCase):
    def test_shape(self):
        m = RNNWithoutInput(7)
        b = asblock(lambda: m(12))
        pred = b.predict()
        print pred
        self.assertEqual(pred.shape, (12, 7))
