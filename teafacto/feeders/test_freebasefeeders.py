from unittest import TestCase
from teafacto.feeders.freebasefeeders import FBLexDataFeedsMaker, getentdict, getglovedict


class TestFreebaseLexFeeder(TestCase):
    def test_getentdic(self):
        d, maxid = getentdict("../../data/freebase/entdic.small.map", top=50)
        self.assertEqual(maxid, 52)
        self.assertEqual(max(d.values()), maxid)

    def test_getglovedict(self):
        d, maxi = getglovedict("../../data/glove/miniglove.50d.txt")
        self.assertEqual(maxi, 4001)
        self.assertEqual(max(d.values()), maxi)

    def test_fb_datafeed(self):
        pass
