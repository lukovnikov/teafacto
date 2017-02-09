from unittest import TestCase
import theano
from theano import tensor as T
import numpy as np

class TestUpdates(TestCase):
    def test_one_loop(self):
        def rec(y):
            return y, {y: y+1}
        x = theano.shared(0)
        o, u = theano.scan(rec, sequences=[], outputs_info=[None],
                           non_sequences=x, n_steps=5)
        f = theano.function([], outputs=[o], updates=u)
        print f()

    def test_nest_loops(self):
        from collections import OrderedDict
        allu = OrderedDict()

        def outerrec(z):
            w = theano.shared(0)
            ao, au = theano.scan(innerrec, sequences=[], outputs_info=[None, None],
                                 non_sequences=[z, w], n_steps=5)
            au[w] = theano.shared(0)
            allu.update(au)
            return ao, au

        def innerrec(y, w):
            return [y, w], {y: y + 1, w: w + 1}

        x = theano.shared(0)
        o, u = theano.scan(outerrec, sequences=[], outputs_info=[None, None],
                           non_sequences=x, n_steps=5)

        allu.update(u)
        f = theano.function([], outputs=o, updates=allu)
        print f()
