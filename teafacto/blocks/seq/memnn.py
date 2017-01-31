##### MEMORY ENABLED RNN's #####
import numpy as np
from teafacto.core.base import Block, Var, Val, param, tensorops as T


# SYMBOLIC OUTPUT MEMORY ENABLED SEQ2SEQ
# - can place attention over all of temporary created output sequence
# - can write to any time step of output (write/erase interface)
# - can do multiple attention steps without actual output (change scalars)

# -> loss is placed over the symbolic output memory
class BulkNN(Block):
    def __init__(self, inpencoder=None,
                memembmat=None, memencoder=None, memsummarizer=None,
                outlen=None, posembdim=None,
                inp_attention=None, out_attention=None,
                nsteps=100, **kw):
        super(BulkNN, self).__init__(**kw)
        self.memembmat = memembmat
        self.memposmat = param((outlen, posembdim), name="outp_posemb").glorotuniform()
        # TODO: position embeddings and support longer sequences during test?
        # OR remove position embeddings completely
        # COULD BE that a separate RNN without any input seq will generate position embeddings better
        # since it can not correlate it with input data
        self.inpencoder = inpencoder
        self.nsteps = nsteps

    def apply(self, inpseq):    # int-(batsize, seqlen)
        inpenc = self.inpencoder(inpseq)    # may carry mask, based on encoder's embedder
        mem_0 = T.zeros(outshape, dtype="float32")  # (batsize, outseqlen, outvocsize)
        h_0 = T.zeros()
        outputs = T.scan(fn=self.rec,
                         outputs_info=[None, h_0, mem_0],
                         n_steps=self.nsteps,
                         non_sequences=inpenc)
        ret = outputs[0]
        return ret

    def rec(self, mem_tm1, h_tm1, inpenc):
        # mem_tm1: f(batsize, outseqlen, outvocsize)
        # h_tm1:   f(batsize, thinkerdim)
        # inpenc:  f(batsize, inplen, inpencdim)

        # summarize memory
        mem_tm1 = self.memnormal(mem_tm1)             # normalize mem values (e.g. softmax)
        mem_tm1_embsum = T.dot(mem_tm1, self.memembmat)  # f(batsize, outseqlen, memembdim)
        mem_tm1_sum = self.memsummar(mem_tm1_embsum)     # f(batsize, outseqlen, memsumdim)
        # TODO: append output position embeddings

        # input and memory read attentions
        inp_ctx_t = self.inp_ctx_block(h_tm1, inpenc)   # (batsize, inpencdim)
        mem_ctx_t = self.out_ctx_block(h_tm1, mem_tm1_sum)  # (batsize, memsumdim)

        # update thinker state
        i_t = T.concatenate([inp_ctx_t, mem_ctx_t], axis=1)
        rnuret = self.thinker.rec(i_t, h_tm1)
        h_t = rnuret[0]

        # memory change interface
        mem_t_addr = self.addr_block(h_t, mem_tm1_sum)  # float-(batsize, outseqlen)
        mem_t_write = self.writevalue_block(h_t)     # (batsize, meminnerdim)
        e_t = self.erase_block(h_t)      # (0..1)-(batsize,)
        c_t = self.change_block(h_t)     # (0..1)-(batsize,)

        # memory change
        can_mem_t = mem_tm1 - T.batched_dot(e_t, mem_tm1 * mem_t_addr.dimshuffle(0, 1, 'x'))    # erase where we addressed
        can_mem_t = can_mem_t + T.batched_tensordot(mem_t_addr, mem_t_write, axes=0)            # write new value
        mem_t = T.batched_dot(1 - c_t, mem_tm1) + T.batched_dot(c_t, can_mem_t)                 # interpolate between old and new value

        return [mem_t, h_t, mem_t]




