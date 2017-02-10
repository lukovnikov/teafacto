from unittest import TestCase

from teafacto.blocks.seq.memnn import BulkNN, SimpleBulkNN, SimpleMemNN
import numpy as np

class MemNNTest(TestCase):
    def test_shapes(self):
        inpvocsize = 5
        inpembdim = 10
        maskid=-1
        posvecdim = 10
        memdim = 12
        memlen = 17
        outdim = 10
        outvocsize = 17

        lastcoredim = outdim + memdim * 3 + posvecdim * 2 + 1 + 1
        coredims = [40, lastcoredim]

        m = SimpleMemNN(inpvocsize=inpvocsize, inpembdim=inpembdim,
                        maskid=maskid, posvecdim=posvecdim,
                        coredims=coredims, memdim=memdim, memlen=memlen,
                        outdim=outdim, outvocsize=outvocsize)

        batsize = 10
        seqlen = 11
        data = np.random.randint(0, inpvocsize, (batsize, seqlen))

        pred = m.predict(data)
        print pred.shape


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
                         nsteps=9,
                         maskid=maskid,
                         memsamplemethod="gumbel",
                         dropout=0.3,
                         memsampletemp=0.3)

        m._return_all_mems = True

        d = np.random.randint(0, inpvocsize, (batsize, seqlen))

        preds = m.predict(d)
        mem_last = preds[0]
        mem_all = preds[1]

        for i in range(mem_all.shape[0]):
            print np.argmax(mem_all[i], axis=2)

        print mem_last.shape
        print mem_all.shape
        self.assertEqual(mem_last.shape, (batsize, memlen, outvocsize))
        self.assertEqual(mem_all.shape, (9, batsize, memlen, outvocsize))