from unittest import TestCase

import numpy as np

from teafacto.blocks.lang.wordembed import Glove


class TestGlove(TestCase):

    def setUp(self):
        self.expshape = (4001, 50)
        Glove.defaultpath = "../../../data/glove/miniglove.%dd.txt"
        self.glove = Glove(self.expshape[1], self.expshape[0]-1)
        print self.glove.defaultpath

    def test_glove(self):
        self.assertEqual(self.glove.W.shape, self.expshape)
        self.assertEqual(self.glove * "the", 1)
        gblock = self.glove.block
        self.assertTrue(np.allclose(self.glove.W, gblock.W.d.get_value()))
        self.assertEqual(self.glove.W.shape, gblock.W.d.get_value().shape)
        self.assertTrue(np.allclose(self.glove % 1, gblock.W.d.get_value()[1, :]))
        gblockpred = gblock.predict([1])
        self.assertTrue(np.allclose(gblockpred, self.glove % "the"))
        self.assertFalse(np.allclose(gblockpred, self.glove % "a"))


