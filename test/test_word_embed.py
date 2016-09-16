from unittest import TestCase

import numpy as np

from teafacto.blocks.lang.wordvec import Glove


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
        self.assertTrue(np.allclose(self.glove.w, gblock.W.d.get_value()))
        self.assertEqual(self.glove.W.shape, gblock.W.d.get_value().shape)
        self.assertTrue(np.allclose(self.glove % 1, gblock.W.d.get_value()[1, :]))
        gblockpred = gblock.predict([1])
        self.assertTrue(np.allclose(gblockpred, self.glove % "the"))
        self.assertFalse(np.allclose(gblockpred, self.glove % "a"))


class TestAdaptedGlove(TestCase):
    def setUp(self):
        wdic = {"the": 10, "a": 5, "his": 50, "abracadabrqmsd--qsdfmqgf-": 6}
        Glove.defaultpath = "../../../data/glove/miniglove.%dd.txt"
        self.glove = Glove(50, 4000).adapt(wdic)
        self.vanillaglove = Glove(50, 4000)

    def test_map(self):
        self.assertEqual(self.glove * "a", 5)
        self.assertEqual(self.glove * "the", 10)
        self.assertEqual(self.glove * "his", 50)
        self.assertEqual(self.glove * "her", 0)
        self.assertEqual(self.glove * "qsdfqlmkdsjfmqlsdkjgmqlsjdf", 0)
        self.assertTrue(np.allclose(self.vanillaglove % "the", self.glove % "the"))

    def test_adapted_block(self):
        gb = self.glove.block
        pred = gb.predict([self.glove * x for x in "the a his".split()])
        vpred = np.asarray([self.vanillaglove % x for x in "the a his".split()])
        self.assertTrue(np.allclose(pred, vpred))
        oovpred = gb.predict([3, 6])    # two different kinds of OOV
        self.assertTrue(np.allclose(oovpred, np.zeros_like(oovpred)))




