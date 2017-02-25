from unittest import TestCase
import numpy as np
from teafacto.core.datafeed import DataFeeder
from teafacto.examples.dummy import Dummy

class TestDataFeeder(TestCase):
    def test_datafeed_random(self):
        for numsam, numbats in [(16, 3), (15, 3), (14, 3), (13, 3), (7, 3)]:
            rd = np.random.randint(0, 5, (numsam, 3))
            df = DataFeeder(rd)
            df.numbats(numbats)
            df.reset()
            ret = []
            while df.hasnextbatch():
                ret.append(df.nextbatch()[0])
            ret = np.concatenate(ret, axis=0)
            print ret.shape
            self.assertEqual(ret.shape, rd.shape)
            self.assertFalse(np.allclose(ret, rd))
            df.reset()
            ret2 = []
            while df.hasnextbatch():
                ret2.append(df.nextbatch()[0])
            ret2 = np.concatenate(ret2, axis=0)
            self.assertEqual(ret2.shape, ret.shape)
            self.assertFalse(np.allclose(ret2, ret))

    def test_dummy_valid_datafeed_randomness(self):
        d = Dummy()
        numsam = 7
        numbats = 3
        data = np.random.randint(0, 5, (numsam, 3))

        ret = [[]]
        def v(a, b):
            ret[0].append(a)
            print np.concatenate(ret[0], axis=0)
            return 0

        d.train([data], data).cross_entropy().adagrad()\
            .validate_on([data], data).extvalid(v)\
            .train(numbats=numbats, epochs=3, _skiptrain=True)

