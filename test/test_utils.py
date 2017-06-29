from unittest import TestCase

from teafacto.util import ticktock as TT, argparsify, loadlexidtsv, \
    unstructurize, restructurize, StringMatrix, tokenize
import os

class TestUtils(TestCase):
    def test_ticktock_duration_string(self):
        tt = TT()
        testdata = [
            (1, "1.000 second"),
            (0.5689, "0.569 second"),
            (0.9999, "1.000 second"),
            (59, "59.000 seconds"),
            (59.00001, "59.000 seconds"),
            (59.0005, "59.001 seconds"),
            (60, "1 minute"),
            (60.005, "1 minute"),
            (61, "1 minute, 1 second"),
            (62, "1 minute, 2 seconds"),
            (121, "2 minutes, 1 second"),
            (120, "2 minutes"),
            (3656, "1 hour, 56 seconds"),
            (2*3600, "2 hours"),
            (24*3600+125, "1 day, 2 minutes, 5 seconds"),
            (25*3600+126, "1 day, 1 hour, 2 minutes, 6 seconds"),
            (50*3600, "2 days, 2 hours")
        ]
        for seconds, text in testdata:
            self.assertEqual(text, tt._getdurationstr(seconds))

    def test_argparsify(self):
        def testf(a=1, b="str"):
            pass
        self.assertEqual(argparsify(testf, test="-a 1"), {"a": 1})


class TestIDTSVLoader(TestCase):
    def test_load(self):
        p = os.path.join(os.path.dirname(__file__), "../data/freebase/labelsrevlex.map.id.tsv.sample")
        print p
        gids, charten, fbids = loadlexidtsv(p)
        print gids.shape, charten.shape, fbids.shape
        self.assertEqual(gids.shape, (10000, 10))
        self.assertEqual(charten.shape, (10000, 10, 30))
        self.assertEqual(fbids.shape, (10000,))


class TestFlatNestF(TestCase):
    def test_f(self):
        s = ["a", "b", ["c", "d"], {"e": ["f", ("g", "h")]}]
        n, f = unstructurize(s)
        self.assertEqual(f, "a b c d f g h".split())
        rs = restructurize(n, f)
        self.assertEqual(rs, s)

import numpy as np
class TestStringMatrix(TestCase):
    def test_everything(self):
        tm = StringMatrix(topnwords=20, indicate_start_end=True, maxlen=15)
        tm.add("the quick brown fox jumped over the lazy dog")
        tm.add("all work and no play make jack a dull boy")
        tm.add("to be or not to be")
        tm.add("what is the meaning of life, universe and everything")
        tm.add("what is love, baby don't hurt me, don't hurt me, no more")
        print tm._wordcounts_original
        print tm._maxlen
        print tm._dictionary
        tm.finalize()
        print tm._wordcounts_original
        print tm._dictionary
        print tm.matrix
        print tm.pp(tm.matrix)
        print tm.matrix[0]
        print tm.pp(np.asarray([0,1,2,3,4,5,6,7,8,9,10,11,12]))
        print tm.pp(tm.matrix[0]+5)


class TestTokenize(TestCase):
    def test_tokenize(self):
        s = "what did he do"
        g = s.split()
        t = tokenize(s)
        self.assertEqual(g, t)

    def test_tokenize_apostrophe_s(self):
        s = "what's etna's height"
        g = "what 's etna 's height".split()
        t = tokenize(s)
        self.assertEqual(g, t)

    def test_tokenize_preserve(self):
        s = "what did <E0> do when <E1> killed <E2> in <E3> during <E4>"
        g = s.split()
        t = tokenize(s, preserve_patterns=["<E\d>"])
        self.assertEqual(g, t)

