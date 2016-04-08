from unittest import TestCase
from teafacto.feeders.freebasefeeders import FBLexDataFeedsMaker, getentdict, getglovedict
from teafacto.core.datafeed import DataFeeder
import os, math


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
        gd, gmaxi = getglovedict(os.path.join(os.path.dirname(__file__), "../data/glove/miniglove.50d.txt"))
        ed, emaxid = getentdict(os.path.join(os.path.dirname(__file__), "../data/freebase/entdic.small.map"), top=50)
        dp = os.path.join(os.path.dirname(__file__), "../data/freebase/labelsrevlex.map.sample")
        f = FBLexDataFeedsMaker(dp, gd, ed, numwords=10, numchars=30)
        self.assertEqual(f.worddic, gd)

        trainfeed = f.trainfeed
        self.assertEqual(trainfeed[0:5].shape, (5, 10, 31))

    def test_fb_datafeed_validosplit(self):
        gd, gmaxi = getglovedict(os.path.join(os.path.dirname(__file__), "../data/glove/miniglove.50d.txt"))
        ed, emaxid = getentdict(os.path.join(os.path.dirname(__file__), "../data/freebase/entdic.small.map"), top=50)
        dp = os.path.join(os.path.dirname(__file__), "../data/freebase/labelsrevlex.map.sample")
        f = FBLexDataFeedsMaker(dp, gd, ed, numwords=10, numchars=30)
        self.assertEqual(f.worddic, gd)

        dfeeder = DataFeeder(*([f.trainfeed] + [f.goldfeed]))
        splits = 1
        dfsplit = dfeeder.osplit(split=splits, random=False)
        dfeeds = dfeeder.feeds
        splitfeeds = dfsplit.feeds
        for x, y in zip(dfeeds, splitfeeds):
            self.assertEqual(x.__class__, y.__class__)
            self.assertEqual(x.ndim, y.ndim)
            self.assertEqual(y.shape[0], int(math.ceil(1.*x.shape[0]/splits)))
            for dim in range(1, len(x.shape)):
                self.assertEqual(x.shape[dim], y.shape[dim])


