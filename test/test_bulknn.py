from unittest import TestCase

from teafacto.blocks.seq.memnn import BulkNN, SimpleBulkNN, SimpleMemNN, SimpleTransMemNN
import numpy as np

class MemNNTest(TestCase):
    def test_shapes(self):
        inpvocsize = 5
        inpembdim = 10
        maskid = -1
        posvecdim = 11
        memdim = 7
        memlen = 5
        outdim = 13
        outvocsize = 15

        lastcoredim = outdim + memdim * 3 + posvecdim * 2 + 1 + 1
        coredims = [40, lastcoredim]

        m = SimpleMemNN(inpvocsize=inpvocsize, inpembdim=inpembdim,
                        maskid=maskid, posvecdim=posvecdim,
                        coredims=coredims, memdim=memdim, memlen=memlen,
                        outdim=outdim, outvocsize=outvocsize,
                        addr_sampler="gumbel", _debug=True)
        m._with_all_mems = True

        batsize = 3
        seqlen = 4
        data = np.random.randint(0, inpvocsize, (batsize, seqlen))

        outs, extra = m.predict(data, _extra_outs=["mem_t_addr", "mem_t_write", "e_t", "c_t", "can_mem_t_after_erase", "can_mem_t_after_addition"])
        pred = outs[0]
        mems = outs[1]
        mem_t_addrs = extra["mem_t_addr"]
        mem_t_writes = extra["mem_t_write"]
        can_mem_t_after_erase = extra["can_mem_t_after_erase"]
        can_mem_t_after_addition = extra["can_mem_t_after_addition"]
        e_ts = extra["e_t"]
        c_ts = extra["c_t"]
        print pred.shape
        #np.set_printoptions(precision=6, suppress=True)
        #print mems
        print mems.shape

        for i in range(mems.shape[0] - 1):
            mem_tm1 = mems[i][0]
            e_t = e_ts[i+1][0]
            c_t = c_ts[i+1][0]
            mem_t_addr = mem_t_addrs[i+1][0]
            mem_t_write = mem_t_writes[i+1][0]
            # CODE REPLICATION
            mem_t, can_mem_t_after_erase_com, can_mem_t_after_addition_com = \
                self.memupdate_replication(e_t, c_t, mem_t_addr, mem_t_write, mem_tm1)
            print can_mem_t_after_erase[i+1][0] - can_mem_t_after_erase_com
            self.assertTrue(np.allclose(can_mem_t_after_erase[i+1][0], can_mem_t_after_erase_com))
            print "-"
            print can_mem_t_after_addition[i+1][0] - can_mem_t_after_addition_com
            self.assertTrue(np.allclose(can_mem_t_after_addition[i+1][0], can_mem_t_after_addition_com))
            print "-"
            print mems[i + 1][0] - mem_t
            self.assertTrue(np.allclose(mems[i+1][0], mem_t))
            print "."

    def memupdate_replication(self, e_t, c_t, mem_t_addr, mem_t_write, mem_tm1):
        can_mem_t = mem_tm1
        can_mem_t = can_mem_t \
                    - e_t * can_mem_t * mem_t_addr[:, np.newaxis]
        can_mem_t_after_erase = can_mem_t
        can_mem_t = can_mem_t + np.outer(mem_t_addr, mem_t_write)
        mem_t = (1 - c_t) * mem_tm1 + c_t * can_mem_t
        return mem_t, can_mem_t_after_erase, can_mem_t


class TransMemNNTest(TestCase):
    def test_shapes(self):
        inpvocsize = 5
        inpembdim = 10
        maskid = -1
        posvecdim = 11
        memdim = 7
        memlen = 5
        outdim = 13
        outvocsize = 15

        lastcoredim = outdim + memdim * 3 + posvecdim * 2 + 1 + 1
        coredims = [40, lastcoredim]

        m = SimpleTransMemNN(inpvocsize=inpvocsize, inpembdim=inpembdim,
                        maskid=maskid, posvecdim=posvecdim,
                        outembdim=inpembdim, outvocsize=outvocsize,
                        coredims=coredims, memdim=memdim, memlen=memlen,
                        outdim=outdim)
        m._with_all_mems = True

        batsize = 3
        seqlen = 4
        inpdata = np.random.randint(1, inpvocsize, (batsize, seqlen))
        outdata = np.concatenate([np.zeros((inpdata.shape[0],))[:, np.newaxis].astype("int32"),
                                          inpdata[:, :-1]], axis=1)
        pred = m.predict(inpdata, outdata)
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