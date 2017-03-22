from unittest import TestCase
from teafacto.blocks.seq.memnn import AutoMorph
from teafacto.blocks.basic import VectorEmbed, SMOWrap
from teafacto.core import Val
import numpy as np
from theano import tensor as T


class TestAutoMorph(TestCase):
    def test_shapes(self):
        charemb = VectorEmbed(256, 20, maskid=0)
        am = AutoMorph(memlen=1000, memkeydim=30, memvaldim=30,
                      charemb=charemb, outdim=32)
        m = SMOWrap(am, outdim=256, inneroutdim=32)
        batsize = 15
        seqlen = 60
        data = np.random.randint(0, 256, (batsize, seqlen))
        data = Val(data)
        pred = m(data)
        predval = pred.eval()
        self.assertEqual(predval.shape, (batsize, seqlen, 256))
        gold = Val(np.random.randint(0, 256, (batsize, seqlen))) + 0

        cost = pred.reshape((-1, pred.shape[-1])) \
            [np.arange(0, batsize*seqlen), gold.reshape((-1,))]
        cost = (-T.log(cost.d)).sum()
        print np.linalg.norm(T.grad(cost, m.outl.W.d).eval())
        print np.linalg.norm(T.grad(cost, m.inner.mem_val.d).eval())
        print np.linalg.norm(T.grad(cost, m.inner.mem_key.d).eval())
        print np.linalg.norm(T.grad(cost, charemb.W.d).eval())



