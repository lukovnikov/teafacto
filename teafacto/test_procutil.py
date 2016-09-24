from unittest import TestCase
from teafacto.procutil import *

class TestProcUtil(TestCase):
    def test_wordmat2charmat(self):
        x = "this is a test"
        wdic = dict(zip(x.split(), range(len(x.split()))))
        xids = np.asarray([wdic[xe] for xe in x.split()]).astype("int32")
        xids = np.expand_dims(xids, axis=0)
        xids = np.concatenate([xids, -1 * np.ones_like(xids)], axis=1)
        charids = wordmat2charmat(xids, wdic)
        self.assertEqual(x, charids2string(charids[0]))

    def test_wordmat2chartensor(self):
        x = "this is a test"
        wdic = dict(zip(x.split(), range(len(x.split()))))
        xids = np.asarray([wdic[xe] for xe in x.split()]).astype("int32")
        xids = np.expand_dims(xids, axis=0)
        xids = np.concatenate([xids, -1 * np.ones_like(xids)], axis=1)
        charten = wordmat2chartensor(xids, wdic)
        self.assertEqual(wordcharmat2string(charten[0]), x)

    def test_charids2string(self):
        x = "this is a test"
        s = charids2string([ord(xe) for xe in x])
        self.assertEqual(s, x)

    def test_wordids2string(self):
        x = "this is a test"
        wdic = dict(zip(x.split(), range(len(x.split()))))
        rwd = {v: k for k, v in wdic.items()}
        s = wordids2string([wdic[xe] for xe in x.split()], rwd)
        self.assertEqual(x, s)
