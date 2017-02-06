from unittest import TestCase

from teafacto.blocks.seq.memnn import BulkNN, SimpleBulkNN
import numpy as np


class BulkNNTest(TestCase):
    def test_init_mem_shape(self):
        inpvocsize = 5
        outvocsize = 7
        inpembdim = 10
        memembdim = 12
        inpdim = [15]
        memdim = [15]
        memlen = 17
        writedim = 19
        lastcoredim = inpdim[-1] + memdim[-1] + memdim[-1] \
                      + writedim + 1 + 1
        coredims = [40, lastcoredim]     # last dim must match interface when explicit interface

        batsize = 13
        seqlen = 11

        m = SimpleBulkNN(inpvocsize=inpvocsize,
                         inpembdim=inpembdim,
                         inpencinnerdim=inpdim,
                         memvocsize=outvocsize,
                         memembdim=memembdim,
                         memencinnerdim=memdim,
                         memlen=memlen,
                         coredims=coredims,
                         explicit_interface=True,
                         write_value_dim=writedim)

        d = np.random.randint(0, inpvocsize, (batsize, seqlen))
        '''
        pred, extras = m.predict(d, _extra_outs=["mem_0", "h_0"])
        print extras["mem_0"].shape
        print pred.shape
        self.assertEqual(pred.shape, (batsize, memlen, outvocsize))
        self.assertEqual(extras["mem_0"].shape, (batsize, memlen, outvocsize))
        '''