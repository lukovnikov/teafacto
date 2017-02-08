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
        posvecdim = 10
        lastcoredim = inpdim[-1] + memdim[-1] + memdim[-1] \
                      + writedim + 1 + 1 + posvecdim * 3
        coredims = [40, lastcoredim]     # last dim must match interface when explicit interface

        batsize = 13
        seqlen = 11

        maskid = 0

        m = SimpleBulkNN(inpvocsize=inpvocsize,
                         inpembdim=inpembdim,
                         inpencinnerdim=inpdim,
                         memvocsize=outvocsize,
                         memembdim=memembdim,
                         memencinnerdim=memdim,
                         memlen=memlen,
                         coredims=coredims,
                         explicit_interface=True,
                         write_value_dim=writedim,
                         posvecdim=posvecdim,
                         nsteps=99,
                         maskid=maskid)

        d = np.random.randint(0, inpvocsize, (batsize, seqlen))

        preds, extras = m.predict(d, _extra_outs=["mem_0", "h_0"])
        mem_last = preds[0]
        mem_all = preds[1]

        print mem_last.shape
        print extras["mem_0"].shape
        print mem_all.shape
        self.assertEqual(mem_last.shape, (batsize, memlen, outvocsize))
        self.assertEqual(extras["mem_0"].shape, (batsize, memlen, outvocsize))
        self.assertEqual(mem_all.shape, (99, batsize, memlen, outvocsize))