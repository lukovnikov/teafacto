from unittest import TestCase

import numpy as np

from teafacto.blocks.embed import Glove


class TestGlove(TestCase):

    def setUp(self):
        self.glove = Glove(4002, 50, test=True)

    def test_glove(self):
        self.assertEqual(self.glove * "the", 1)
        gblock = self.glove.block
        self.assertTrue(np.allclose(self.glove.W, gblock.W.d.get_value()))
        self.assertEqual(self.glove.W.shape, gblock.W.d.get_value().shape)
        self.assertTrue(np.allclose(self.glove % 1, gblock.W.d.get_value()[1, :]))
        gblockpred = gblock.predict([1])
        self.assertTrue(np.allclose(gblockpred, self.glove % "the"))
        self.assertFalse(np.allclose(gblockpred, self.glove % "a"))
        self.assertTrue(np.allclose(self.glove % 4001, np.zeros_like(self.glove % 4001)))


