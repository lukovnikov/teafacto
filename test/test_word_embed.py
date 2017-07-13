from unittest import TestCase

import numpy as np

from teafacto.blocks.word.wordvec import Glove, WordEmb
from teafacto.blocks.basic import VectorEmbed
from teafacto.core.base import Val


class TestVectorEmbed(TestCase):
    def test_embed_masker(self):
        x = VectorEmbed(indim=5, dim=5, maskid=0)
        v = Val(np.random.randint(0, 5, (4, 3)))
        xo = x(v)
        self.assertEqual(v.v.shape, xo.mask.v.shape)
        self.assertTrue(np.all((v.v != 0) == xo.mask.v))


class TestGlove(TestCase):

    def setUp(self):
        numwords = 4000
        self.expshape = (4002, 50)
        Glove.defaultpath = "../../../data/glove/miniglove.%dd"
        self.glove = Glove(self.expshape[1], numwords)
        print self.glove.defaultpath

    def test_glove_embed_masker(self):
        v = Val(np.random.randint(0, 7, (40, 30)))
        self.assertTrue(np.all((v.v != 0) == self.glove(v).mask.v))

    def test_glove(self):
        self.assertEqual(self.glove.w.shape, self.expshape)
        self.assertEqual(self.glove * "the", 2)
        gblock = self.glove.block
        self.assertTrue(np.allclose(self.glove.w, gblock.W.d.get_value()))
        self.assertEqual(self.glove.w.shape, gblock.W.d.get_value().shape)
        self.assertTrue(np.allclose(self.glove % 2, gblock.W.d.get_value()[2, :]))
        gblockpred = gblock.predict([2])
        self.assertTrue(np.allclose(gblockpred, self.glove % "the"))
        self.assertFalse(np.allclose(gblockpred, self.glove % "a"))


class TestAdaptedGlove(TestCase):
    def setUp(self):
        wdic = {"<MASK>": 0, "<RARE>": 1, "the": 10, "a": 5, "his": 50, "abracadabrqmsd--qsdfmqgf-": 6}
        Glove.defaultpath = "../../../data/glove/miniglove.%dd"
        self.glove = Glove(50, 4000).adapt(wdic)
        self.vanillaglove = Glove(50, 4000)

    def test_embed_masker(self):
        v = Val(np.random.randint(0, 5, (4, 3)))
        self.assertTrue(np.all((v.v != 0) == self.glove(v).mask.v))

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


class TestGloveOverriding(TestCase):
    def setUp(self):
        words = "<MASK> <RARE> the a his monkey inception key earlgrey"
        wdic = dict(zip(words.split(), range(0, len(words.split()))))
        self.baseemb = WordEmb(dim=50, worddic=wdic)
        Glove.defaultpath = "../../../data/glove/miniglove.%dd"
        self.glove = Glove(50, 4000)
        self.emb = self.baseemb.override(self.glove)
        pass

    def test_embed_masker(self):
        v = Val(np.random.randint(0, 5, (4, 3)))
        m = self.emb(v).mask
        self.assertTrue(np.all((v.v != 0) == m.v))

    def test_sameasglove(self):
        words = "key the a his"
        pred = self.emb.predict([self.emb * x for x in words.split()])
        gpred = self.glove.predict([self.glove * x for x in words.split()])
        self.assertTrue(np.allclose(pred, gpred))

    def test_sameasbase(self):
        words = "inception monkey earlgrey"
        pred = self.emb.predict([self.emb * x for x in words.split()])
        gpred = self.baseemb.predict([self.baseemb * x for x in words.split()])
        self.assertTrue(np.allclose(pred, gpred))

    def test_notasglove(self):
        words = "inception monkey earlgrey"
        pred = self.emb.predict([self.emb * x for x in words.split()])
        gpred = self.glove.predict([self.glove * x for x in words.split()])
        self.assertFalse(np.allclose(pred, gpred))

    def test_notasbase(self):
        words = "key the a his"
        pred = self.emb.predict([self.emb * x for x in words.split()])
        gpred = self.baseemb.predict([self.baseemb * x for x in words.split()])
        self.assertFalse(np.allclose(pred, gpred))


class TestAugmentedGlove(TestCase):
    def test_gloveglove(self):
        Glove.defaultpath = "../../../data/glove/miniglove.%dd"
        g1 = Glove(50, 2000)
        g2 = Glove(50, 1000)
        gloveglove = g1.augment(g2)
        pred = gloveglove.predict([1002])
        self.assertTrue(np.allclose(g1 % 1002, pred[0, :50]))
        self.assertTrue(np.allclose(g2 % 1002, pred[0, 50:]))
        pred = gloveglove.predict([1003])
        self.assertTrue(np.allclose(g1 % 1003, pred[0, :50]))
        self.assertTrue(np.allclose(pred[0, 50:], np.zeros_like(pred[0, 50:])))
        gloveglove = g2.augment(g1)
        pred = gloveglove.predict([1, 2, 3, 4, 5, 50, 500, 1000, 1001])
        self.assertTrue(np.allclose(pred[:, :50], pred[:, 50:]))

