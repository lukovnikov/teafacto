##### MEMORY ENABLED RNN's #####
import numpy as np
from teafacto.core.base import Block, Var, Val, param, tensorops as T

# TODO: INPUT MASK !!!!!!!! and attention etc


# SYMBOLIC OUTPUT MEMORY ENABLED SEQ2SEQ
# - can place attention over all of temporary created output sequence
# - can write to any time step of output (write/erase interface)
# - can do multiple attention steps without actual output (change scalars)

# -> loss is placed over the symbolic output memory
class BulkNN(Block):
    def __init__(self, inpencoder=None, memsampler=None,
                memembmat=None, memencoder=None, memlen=None,
                posembdim=None,
                inp_attention=None, mem_attention=None,
                inp_addr_extractor=None, mem_addr_extractor=None,
                write_addr_extractor=None, write_addr_attention=None,
                write_value_generator=None,
                mem_erase_generator=None, mem_change_generator=None,
                nsteps=100, core=None, **kw):
        super(BulkNN, self).__init__(**kw)
        self._memposemb = None
        if posembdim is not None:
            self._memposemb = param((memlen, posembdim), name="outp_posemb").glorotuniform()
        self._nsteps = nsteps
        self._memlen = memlen
        self._inpencoder = inpencoder
        self._inp_att = inp_attention
        self._memencoder = memencoder
        self._mem_att = mem_attention
        self._memembmat = memembmat
        self._memsampler = memsampler
        self._core = core
        # extractors from top core state:
        self._inp_addr_extractor = inp_addr_extractor
        self._mem_addr_extractor = mem_addr_extractor
        self._write_addr_extractor = write_addr_extractor
        self._write_addr_attention = write_addr_attention
        self._write_value_generator = write_value_generator
        self._mem_change_generator = mem_change_generator
        self._mem_erase_generator = mem_erase_generator

    def apply(self, inpseq):    # int-(batsize, seqlen)
        inpenc = self._inpencoder(inpseq)    # may carry mask, based on encoder's embedder
        batsize = inpenc.shape[0]
        outvocsize = self._memembmat.shape[0]
        mem_0 = T.concatenate([
            T.ones((batsize, self._memlen, 1), dtype="float32") * 0.95,
            T.ones((batsize, self._memlen, outvocsize-1), dtype="float32") * 0.05,
            ], axis=2)      # (batsize, outseqlen, outvocsize)
        mem_0 = T.softmax(mem_0)
        core_init_states = self._core.get_init_info(batsize)
        core_state_spec = self._core.get_statespec()
        assert(len(core_state_spec) == len(core_init_states))
        h_0 = None  # take last output of core states as initial state
        for i in range(len(core_state_spec)):
            ss = core_state_spec[i]
            if ss[0] == "output":
                h_0 = core_init_states[i]
        outputs = T.scan(fn=self.rec,
                         outputs_info=[None, mem_0, h_0] + core_init_states,
                         n_steps=self._nsteps,
                         non_sequences=inpenc)
        ret = outputs[0][-1]
        ret.push_extra_outs({"mem_0": mem_0, "h_0": h_0})   # DEBUGGING
        return ret

    def rec(self, mem_tm1, h_tm1, *args):
        inpenc = args[-1]
        states_tm1 = args[:-1]
        #return (mem_tm1, mem_tm1, h_tm1) + states_tm1   # DEBUG
        # mem_tm1: f(batsize, outseqlen, outvocsize)
        # h_tm1:   f(batsize, thinkerdim)
        # inpenc:  f(batsize, inplen, inpencdim)

        # summarize memory
        mem_tm1_sam = self._memsample(mem_tm1)               # sample from mem
        mem_tm1_embsum = T.dot(mem_tm1_sam, self._memembmat)  # f(batsize, outseqlen, memembdim)
        mem_tm1_sum = self._memencode(mem_tm1_embsum)     # f(batsize, outseqlen, memsumdim)
        # TODO: append output position embeddings

        # input and memory read attentions
        inp_ctx_t = self._get_inp_ctx(h_tm1, inpenc)   # (batsize, inpencdim)
        mem_ctx_t = self._get_mem_ctx(h_tm1, mem_tm1_sum)  # (batsize, memsumdim)

        # update thinker state
        i_t = T.concatenate([inp_ctx_t, mem_ctx_t], axis=1)
        rnuret = self._core.rec(i_t, *states_tm1)
        h_t = rnuret[0]
        states_t = rnuret[1:]

        # memory change interface
        mem_t_addr = self._get_addr_weights(h_t, mem_tm1_sum)  # float-(batsize, outseqlen)
        mem_t_write = self._get_write_weights(h_t)     # (batsize, memvocsize)
        e_t = self._get_erase(h_t)      # (0..1)-(batsize,)
        c_t = self._get_change(h_t)     # (0..1)-(batsize,)

        # memory change
        can_mem_t = mem_tm1 - T.batched_dot(e_t, mem_tm1 * mem_t_addr.dimshuffle(0, 1, 'x'))    # erase where we addressed
        can_mem_t = can_mem_t + T.batched_tensordot(mem_t_addr, mem_t_write, axes=0)            # write new value
        mem_t = T.batched_dot(1 - c_t, mem_tm1) + T.batched_dot(c_t, can_mem_t)                 # interpolate between old and new value

        mem_t = T.softmax(mem_t)        # normalize to probabilities
        return (mem_t, mem_t, h_t) + states_t

    def _memsample(self, mem):
        if self._memsampler is None:
            return mem
        else:
            return self._memsampler(mem)

    def _memencode(self, mem):
        if self._memencoder is None:
            return mem
        else:
            return self._memencoder(mem)

    def _get_inp_ctx(self, h, inpenc):
        crit = self._inp_addr_extractor(h)
        return self._inp_att(crit, inpenc)

    def _get_mem_ctx(self, h, mem):
        crit = self._mem_addr_extractor(h)
        return self._mem_att(crit, mem)

    def _get_addr_weights(self, h, mem):
        crit = self._write_addr_extractor(h)
        return self._write_addr_generator(crit, mem)

    def _get_write_weights(self, h):
        return self._write_value_generator(h)  # generate categorical write distr

    def _get_erase(self, h):
        return self._mem_erase_generator(h)

    def _get_change(self, h):
        return self._mem_change_generator(h)


from teafacto.blocks.seq.rnn import SeqEncoder, MakeRNU
from blocks.seq.recstack import RecStack
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.match import CosineDistance
from teafacto.blocks.seq.attention import Attention, AttGen


class SimpleBulkNN(BulkNN):
    """ Parameterized simple interface for BulkNN that builds defaults for subcomponents """
    def __init__(self, inpvocsize=None, inpembdim=None, inpemb=None,
                        inpencinnerdim=None, bidir=False, maskid=None,
                        dropout=False, rnu=GRU, inpencoder=None,
                       memvocsize=None, memembdim=None, memembmat=None,
                        memencinnerdim=None, memencoder=None,
                       inp_att_dist=CosineDistance(), mem_att_dist=CosineDistance(),
                        inp_attention=None, mem_attention=None,
                       coredims=None, corernu=GRU, core=None,
                 **kw):
        # INPUT ENCODING
        if inpencoder is None:
            inpencoder = SeqEncoder.RNN(indim=inpvocsize, inpembdim=inpembdim,
                        inpemb=inpemb, innerdim=inpencinnerdim, bidir=bidir,
                        maskid=maskid, dropout_in=dropout, dropout_h=dropout,
                        rnu=rnu).all_outputs()
        # MEMORY ENCODING
        if memembmat is None:
            memembmat = param((memvocsize, memembdim), name="memembmat").glorotuniform()
        if memencoder is None:
            memencoder = SeqEncoder.RNN(inpemb=False, innerdim=memencinnerdim,
                        bidir=bidir, dropout_in=dropout, dropout_h=dropout,
                        rnu=rnu).all_outputs()
        # TWO READ ATTENTIONS
        if inp_attention is None:
            inp_attention = Attention(inp_att_dist)
        if mem_attention is None:
            mem_attention = Attention(mem_att_dist)

        # CORE RNN - THE THINKER
        if core is None:
            corelayers, _ = MakeRNU.fromdims([inpencinnerdim+memencinnerdim] + coredims,
                                    rnu=corernu, dropout_in=dropout, dropout_h=dropout)
            core = RecStack(*corelayers)

        # WRITE INTERFACE

        # MEM UPDATE INTERFACE

        super(SimpleBulkNN, self).__init__(inpencoder=inpencoder,
            memembmat=memembmat, memencoder=memencoder,
            inp_attention=inp_attention, mem_attention=mem_attention,
            core=core,
            **kw)

