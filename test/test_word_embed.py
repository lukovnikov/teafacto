from unittest import TestCase
from teafacto.blocks.lm import Glove
import numpy as np


class TestGlove(TestCase):

    def setUp(self):
        self.glove = Glove(4000, 50)

    def test_glove(self):
        self.assertEqual(self.glove * "the", 1)
        gblock = self.glove.block
        self.assertTrue(np.allclose(self.glove.W, gblock.W.d.get_value()))
        self.assertEqual(self.glove.W.shape, gblock.W.d.get_value().shape)
        self.assertTrue(np.allclose(self.glove % 1, gblock.W.d.get_value()[1, :]))
        gblockpred = gblock.predict([1])
        self.assertTrue(np.allclose(gblockpred, self.glove % "the"))


