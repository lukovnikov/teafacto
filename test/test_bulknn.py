from unittest import TestCase

from teafacto.blocks.seq.memnn import BulkNN, SimpleBulkNN
import numpy as np


class BulkNNTest(TestCase):
    def test_init_mem_shape(self):
        inpvocsize = 5
        outvocsize = 7
        impembdim = 10
        memembdim = 12
        inpdim = 15
        memdim = 15
        memlen = 17
        coredims = [20, 30]

        batsize = 13
        seqlen = 11

        m = SimpleBulkNN(inpvocsize=inpvocsize, inpembdim=impembdim, inpencinnerdim=inpdim,
                         memvocsize=outvocsize, memembdim=memembdim, memencinnerdim=memdim,
                         memlen=memlen, coredims=coredims)

        d = np.random.randint(0, inpvocsize, (batsize, seqlen))

        pred, extras = m.predict(d, _extra_outs=["mem_0", "h_0"])
        #print extras["mem_0"].shape
        print pred.shape
        self.assertEqual(pred.shape, (batsize, memlen, outvocsize))
        self.assertEqual(extras["mem_0"].shape, (batsize, memlen, outvocsize))