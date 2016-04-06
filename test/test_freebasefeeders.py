from unittest import TestCase
from teafacto.feeders.freebasefeeders import FBLexDataFeedsMaker, getentdict, getglovedict
import os


class TestFreebaseLexFeeder(TestCase):
    def test_getentdic(self):
        d, maxid = getentdict(os.path.join(os.path.dirname(__file__), "../data/freebase/entdic.small.map"), top=50)
        self.assertEqual(maxid, 52)
        self.assertEqual(max(d.values()), maxid)

    def test_getglovedict(self):
        d, maxi = getglovedict(os.path.join(os.path.dirname(__file__), "../data/glove/miniglove.50d.txt"))
        self.assertEqual(maxi, 4001)
        self.assertEqual(max(d.values()), maxi)

    def test_fb_datafeed(self):
        pass