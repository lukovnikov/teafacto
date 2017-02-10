from teafacto.core.base import Block, Var, Val, RVal, param, tensorops as T
from IPython import embed
import numpy as np


class MemNN(Block):
    """ Basic key-value store """
    def __init__(self, memlen=None, core=None, mem_pos_repr=None,
                 mem_attention=None, mem_addr_extractor=None,
                write_addr_extractor=None, write_addr_generator=None,
                write_value_extractor=None, outvec_extractor=None,
                mem_erase_generator=None, mem_change_generator=None,
                inpemb=None, memdim=None, smo=None,
                 **kw):
        super(MemNN, self).__init__(**kw)
        self._memlen = memlen
        self._core = core
        if mem_pos_repr is not None:
            self._memposvecs = mem_pos_repr(memlen)
        else:
            self._memposvecs = None
        self._mem_addr_extractor = mem_addr_extractor
        self._mem_att = mem_attention
        self._write_addr_extractor = write_addr_extractor
        self._write_addr_generator = write_addr_generator
        self._write_value_extractor = write_value_extractor
        self._mem_change_generator = mem_change_generator
        self._mem_erase_generator = mem_erase_generator

        self.inpembedder = inpemb
        self.memdim = memdim
        self.smo = smo
        self.outvec_extractor = outvec_extractor

    def apply(self, inpseq):  # int-(batsize, seqlen)
        inpemb = self.inpembedder(inpseq)  # may carry mask, based on encoder's embedder
        batsize = inpseq.shape[0]

        mem_0 = Val(np.zeros((1, self._memlen, self.memdim)))
        mem_0 = T.repeat(mem_0, batsize, axis=0)

        core_init_states = self._core.get_init_info(batsize)
        core_state_spec = self._core.get_statespec(flat=False)
        assert (len(core_state_spec) == len(core_init_states))
        h_0 = None  # take last output of core states as initial state
        c = 0
        for ss in core_state_spec:
            h_0_isout = False
            for sss in ss:
                if sss[0] == "output":
                    h_0_isout = True
                    h_0 = core_init_states[c]
                if not h_0_isout:
                    h_0 = core_init_states[c]
                c += 1
        recinp = inpemb.dimswap(1, 0)
        outputs = T.scan(fn=self.rec,
                         sequences=recinp,
                         outputs_info=[None, mem_0, h_0] + core_init_states)
        ret = outputs[0].dimswap(1, 0)
        return ret

    def rec(self, x_t, mem_tm1, h_tm1, *args):
        states_tm1 = args
        batsize = x_t.shape[0]

        mem_tm1_sum = mem_tm1
        if self._memposvecs is not None:
            memposvecs = T.repeat(self._memposvecs.dimadd(0), batsize, axis=0)
            mem_tm1_sum = T.concatenate([mem_tm1_sum, memposvecs], axis=2)

        # input and memory read attentions
        mem_ctx_t = self._get_mem_ctx(h_tm1, mem_tm1_sum)  # (batsize, memsumdim)

        # update thinker state
        i_t = T.concatenate([x_t, mem_ctx_t], axis=1)
        rnuret = self._core.rec(i_t, *states_tm1)
        h_t = rnuret[0]
        states_t = rnuret[1:]

        # memory change interface
        mem_t_addr = self._get_addr_weights(h_t, mem_tm1_sum)  # float-(batsize, outseqlen)
        mem_t_write = self._get_write_weights(h_t)  # (batsize, memvocsize)
        e_t = self._get_erase(h_t)  # (0..1)-(batsize,)
        c_t = self._get_change(h_t)  # (0..1)-(batsize,)

        # e_t = T.zeros_like(e_t)     # DEBUG

        # memory change
        can_mem_t = mem_tm1
        can_mem_t = can_mem_t - T.batched_dot(e_t,
                        can_mem_t * mem_t_addr.dimshuffle(0, 1, 'x'))  # erase where we addressed
        can_mem_t = can_mem_t + T.batched_tensordot(mem_t_addr, mem_t_write, axes=0)  # write new value
        mem_t = T.batched_dot(1 - c_t, mem_tm1) + T.batched_dot(c_t, can_mem_t)  # interpolate between old and new value

        _y_t = self.outvec_extractor(h_t)
        y_t = self.smo(_y_t)
        return (y_t, mem_t, h_t) + tuple(states_t)

    def _get_mem_ctx(self, h, mem):
        crit = self._mem_addr_extractor(h)
        return self._mem_att(crit, mem)

    def _get_addr_weights(self, h, mem):
        crit = self._write_addr_extractor(h)
        return self._write_addr_generator(crit, mem)

    def _get_write_weights(self, h):
        crit = self._write_value_extractor(h)
        return crit  # generate categorical write distr

    def _get_erase(self, h):
        # return T.sum(self._mem_erase_generator(h), axis=1)
        ret = self._mem_erase_generator(h)
        ret = T.nnet.sigmoid(ret)
        return ret

    def _get_change(self, h):
        # return T.sum(self._mem_change_generator(h), axis=1)
        ret = self._mem_change_generator(h)
        ret = T.nnet.sigmoid(ret)
        return ret


# SYMBOLIC OUTPUT MEMORY ENABLED SEQ2SEQ
# - can place attention over all of temporary created output sequence
# - can write to any time step of output (write/erase interface)
# - can do multiple attention steps without actual output (change scalars)

# -> loss is placed over the symbolic output memory
class BulkNN(MemNN):
    def __init__(self, inpencoder=None, memsampler=None,
                memembmat=None, memencoder=None,
                inp_pos_repr=None,
                inp_attention=None,
                inp_addr_extractor=None,
                 write_value_generator=None,
                nsteps=100, **kw):
        super(BulkNN, self).__init__(**kw)
        self._inp_pos_repr = inp_pos_repr
        self._nsteps = nsteps
        self._inpencoder = inpencoder
        self._inp_att = inp_attention
        self._memencoder = memencoder
        self._memembmat = memembmat
        self._memsampler = memsampler
        # extractors from top core state:
        self._inp_addr_extractor = inp_addr_extractor
        self._return_all_mems = False
        self._write_value_generator = write_value_generator

    def apply(self, inpseq):    # int-(batsize, seqlen)
        inpenco = self._inpencoder(inpseq)    # may carry mask, based on encoder's embedder
        batsize = inpenco.shape[0]
        outvocsize = self._memembmat.shape[0]
        mem_0 = T.concatenate([
            T.ones((batsize, self._memlen, 1), dtype="float32") * 0.95,
            T.ones((batsize, self._memlen, outvocsize-1), dtype="float32") * 0.05,
            ], axis=2)      # (batsize, outseqlen, outvocsize)

        # DEBUG
        #mem_0 = Val(np.random.random((1, self._memlen, outvocsize)))
        #mem_0 = T.repeat(mem_0, batsize, axis=0)

        mem_0 = T.softmax(mem_0)
        core_init_states = self._core.get_init_info(batsize)
        core_state_spec = self._core.get_statespec(flat=False)
        assert(len(core_state_spec) == len(core_init_states))
        h_0 = None  # take last output of core states as initial state
        c = 0
        for ss in core_state_spec:
            h_0_isout = False
            for sss in ss:
                if sss[0] == "output":
                    h_0_isout = True
                    h_0 = core_init_states[c]
                if not h_0_isout:
                    h_0 = core_init_states[c]
                c += 1
        if self._inp_pos_repr is not None:
            inpposvecs = self._inp_pos_repr(inpseq.shape[1])
            inpposvecs = T.repeat(inpposvecs.dimadd(0), batsize, axis=0)
            inpenc = T.concatenate([inpenco, inpposvecs], axis=2)
            inpenc.mask = inpenco.mask
        else:
            inpenc = inpenco
        outputs = T.scan(fn=self.rec,
                         outputs_info=[None, mem_0, h_0] + core_init_states,
                         n_steps=self._nsteps,
                         non_sequences=inpenc)
        ret = outputs[0]
        if self._return_all_mems:
            return ret[-1], ret
        else:
            return ret[-1]

    def rec(self, mem_tm1, h_tm1, *args):
        inpenc = args[-1]
        states_tm1 = args[:-1]
        batsize = inpenc.shape[0]
        #return (mem_tm1, mem_tm1, h_tm1) + states_tm1   # DEBUG
        # mem_tm1: f(batsize, outseqlen, outvocsize)
        # h_tm1:   f(batsize, thinkerdim)
        # inpenc:  f(batsize, inplen, inpencdim)

        # summarize memory
        mem_tm1_sam = self._memsample(mem_tm1)               # sample from mem
        mem_tm1_embsum = T.dot(mem_tm1_sam, self._memembmat)  # f(batsize, outseqlen, memembdim)
        mem_tm1_sum = self._memencode(mem_tm1_embsum)     # f(batsize, outseqlen, memsumdim)

        if self._memposvecs is not None:
            memposvecs = T.repeat(self._memposvecs.dimadd(0), batsize, axis=0)
            mem_tm1_sum = T.concatenate([mem_tm1_sum, memposvecs], axis=2)

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

        #e_t = T.zeros_like(e_t)     # DEBUG

        # memory change
        can_mem_t = mem_tm1
        can_mem_t = can_mem_t - T.batched_dot(e_t, can_mem_t * mem_t_addr.dimshuffle(0, 1, 'x'))    # erase where we addressed
        can_mem_t = can_mem_t + T.batched_tensordot(mem_t_addr, mem_t_write, axes=0)            # write new value
        mem_t = T.batched_dot(1 - c_t, mem_tm1) + T.batched_dot(c_t, can_mem_t)                 # interpolate between old and new value

        mem_t = T.softmax(mem_t)        # normalize to probabilities
        return (mem_t, mem_t, h_t) + tuple(states_t)

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

    def _get_write_weights(self, h):
        crit = self._write_value_extractor(h)
        return self._write_value_generator(crit)  # generate categorical write distr


from teafacto.blocks.seq.rnn import SeqEncoder, MakeRNU, RecStack, RNNWithoutInput
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.match import CosineDistance
from teafacto.blocks.seq.attention import Attention, AttGen
from teafacto.blocks.basic import MatDot, Linear, Forward, SMO, VectorEmbed
from teafacto.blocks.activations import GumbelSoftmax
from teafacto.core.base import asblock
from teafacto.util import issequence


class SimpleMemNN(MemNN):
    def __init__(self, inpvocsize=None, inpembdim=None, inpemb=None,
                 maskid=None, dropout=False, rnu=GRU,
                 posvecdim=None, mem_pos_repr=None,
                 core=None, coredims=None,
                 mem_att_dist=CosineDistance(), mem_attention=None,
                    mem_addr_extractor=None,
                     write_addr_extractor=None, write_addr_generator=None,
                     write_addr_dist=CosineDistance(),
                     write_value_extractor=None,
                     mem_erase_generator=None, mem_change_generator=None,
                 memdim=None, memlen=None,
                 smo=None, outdim=None, outvocsize=None, **kw):
        # INPUT SYMBOL EMBEDDER
        if inpemb is None:
            inpemb = VectorEmbed(inpvocsize, inpembdim, maskid=maskid)

        # POSITION VECTORS
        if posvecdim is not None and mem_pos_repr is None:
            #mem_pos_repr = RNNWithoutInput(posvecdim, dropout=dropout)
            mem_pos_repr = MatParamGen(memlen, posvecdim)

        xtra_dim = posvecdim if posvecdim is not None else 0
        # CORE RNN - THE THINKER
        if core is None:
            corelayers, _ = MakeRNU.fromdims([memdim + xtra_dim * 2] + coredims,
                                             rnu=rnu, dropout_in=dropout, dropout_h=dropout,
                                             param_init_states=True)
            core = RecStack(*corelayers)

        lastcoredim = core.get_statespec()[-1][0][1][0]

        # ATTENTIONS
        if mem_attention is None:
            mem_attention = Attention(mem_att_dist)
        if write_addr_generator is None:
            write_addr_generator = AttGen(write_addr_dist)

        if smo is None:
            smo = SMO(outdim, outvocsize)

        outvec_extractor, mem_addr_extractor, write_addr_extractor,\
        write_value_extractor, mem_erase_generator, mem_change_generator = \
            make_vector_slicers(lastcoredim, outdim, memdim + xtra_dim,
                                memdim + xtra_dim, memdim, 1, 1)

        super(SimpleMemNN, self).__init__(memlen=memlen, core=core,
                mem_pos_repr=mem_pos_repr, outvec_extractor=outvec_extractor,
                 mem_attention=mem_attention, mem_addr_extractor=mem_addr_extractor,
                write_addr_extractor=write_addr_extractor, write_addr_generator=write_addr_generator,
                write_value_extractor=write_value_extractor,
                mem_erase_generator=mem_erase_generator, mem_change_generator=mem_change_generator,

                inpemb=inpemb, memdim=memdim, smo=smo, **kw)


class SimpleBulkNN(BulkNN):
    """ Parameterized simple interface for BulkNN that builds defaults for subcomponents """
    def __init__(self, inpvocsize=None, inpembdim=None, inpemb=None,
                        inpencinnerdim=None, bidir=False, maskid=None,
                        dropout=False, rnu=GRU,
                 inpencoder=None,
                       memvocsize=None, memembdim=None, memembmat=None,
                        memencinnerdim=None,
                 memencoder=None,
                       inp_att_dist=CosineDistance(), mem_att_dist=CosineDistance(),
                        inp_attention=None, mem_attention=None,
                       coredims=None, corernu=GRU,
                 core=None, explicit_interface=False, scalaraggdim=None,
                                        write_value_dim=None, nsteps=100,
                 posvecdim=None, mem_pos_repr=None, inp_pos_repr=None,
                     inp_addr_extractor=None, mem_addr_extractor=None,
                     write_addr_extractor=None, write_addr_generator=None,
                     write_addr_dist=CosineDistance(),
                     write_value_generator=None, write_value_extractor=None,
                     mem_erase_generator=None, mem_change_generator=None,
                 memsampler=None, memsamplemethod=None, memsampletemp=0.3,
                 **kw):

        # INPUT ENCODING
        if inpencoder is None:
            inpencoder = SeqEncoder.RNN(indim=inpvocsize, inpembdim=inpembdim,
                        inpemb=inpemb, innerdim=inpencinnerdim, bidir=bidir,
                        maskid=maskid, dropout_in=dropout, dropout_h=dropout,
                        rnu=rnu).all_outputs()
            lastinpdim = inpencinnerdim if not issequence(inpencinnerdim) else inpencinnerdim[-1]
        else:
            lastinpdim = inpencoder.block.layers[-1].innerdim

        # MEMORY ENCODING
        if memembmat is None:
            memembmat = param((memvocsize, memembdim), name="memembmat").glorotuniform()
        if memencoder is None:
            memencoder = SeqEncoder.RNN(inpemb=False, innerdim=memencinnerdim,
                        bidir=bidir, dropout_in=dropout, dropout_h=dropout,
                        rnu=rnu, inpembdim=memembdim).all_outputs()
            lastmemdim = memencinnerdim if not issequence(memencinnerdim) else memencinnerdim[-1]
        else:
            lastmemdim = memencoder.block.layers[-1].innerdim

        # POSITION VECTORS
        if posvecdim is not None and inp_pos_repr is None:
            inp_pos_repr = RNNWithoutInput(posvecdim, dropout=dropout)
        if posvecdim is not None and mem_pos_repr is None:
            mem_pos_repr = RNNWithoutInput(posvecdim, dropout=dropout)

        xtra_dim = posvecdim if posvecdim is not None else 0
        # CORE RNN - THE THINKER
        if core is None:
            corelayers, _ = MakeRNU.fromdims([lastinpdim+lastmemdim+xtra_dim*2] + coredims,
                                    rnu=corernu, dropout_in=dropout, dropout_h=dropout,
                                    param_init_states=True)
            core = RecStack(*corelayers)

        lastcoredim = core.get_statespec()[-1][0][1][0]

        # ATTENTIONS
        if mem_attention is None:
            mem_attention = Attention(mem_att_dist)
        if inp_attention is None:
            inp_attention = Attention(inp_att_dist)
        if write_addr_generator is None:
            write_addr_generator = AttGen(write_addr_dist)

        # WRITE VALUE
        if write_value_generator is None:
            write_value_generator = WriteValGenerator(write_value_dim, memvocsize, dropout=dropout)

        # MEMORY SAMPLER
        if memsampler is not None:
            assert(memsamplemethod is None)
        if memsamplemethod is "gumbel":
            assert(memsampler is None)
            memsampler = GumbelSoftmax(temperature=memsampletemp)

        ################ STATE INTERFACES #################

        if not explicit_interface:
            if inp_addr_extractor is None:
                inp_addr_extractor = Forward(lastcoredim, lastinpdim + xtra_dim, dropout=dropout)
            if mem_addr_extractor is None:
                inp_addr_extractor = Forward(lastcoredim, lastmemdim + xtra_dim, dropout=dropout)

            # WRITE INTERFACE
            if write_addr_extractor is None:
                write_addr_extractor = Forward(lastcoredim, lastmemdim + xtra_dim, dropout=dropout)
            if write_value_extractor is None:
                write_value_extractor = Forward(lastcoredim, write_value_dim, dropout=dropout)

            # MEM UPDATE INTERFACE
            if mem_erase_generator is None:
                mem_erase_generator = StateToScalar(lastcoredim, scalaraggdim)
            if mem_change_generator is None:
                mem_change_generator = StateToScalar(lastcoredim, scalaraggdim)
        else:
            inp_addr_extractor, mem_addr_extractor, write_addr_extractor, \
            write_value_extractor, mem_erase_generator, mem_change_generator = \
                make_vector_slicers(lastcoredim, lastinpdim + xtra_dim, lastmemdim + xtra_dim,
                                    lastmemdim + xtra_dim, write_value_dim, 1, 1)

        super(SimpleBulkNN, self).__init__(inpencoder=inpencoder,
            memembmat=memembmat, memencoder=memencoder,
            inp_attention=inp_attention, mem_attention=mem_attention,
            core=core, memsampler=memsampler, nsteps=nsteps,
            inp_addr_extractor=inp_addr_extractor, mem_addr_extractor=mem_addr_extractor,
            write_addr_extractor=write_addr_extractor, write_addr_generator=write_addr_generator,
            mem_erase_generator=mem_erase_generator, mem_change_generator=mem_change_generator,
            write_value_generator=write_value_generator, write_value_extractor=write_value_extractor,
            inp_pos_repr=inp_pos_repr, mem_pos_repr=mem_pos_repr,
            **kw)


class WriteValGenerator(Block):
    def __init__(self, dim, vocsize, interdims=tuple(), dropout=False, **kw):
        super(WriteValGenerator, self).__init__(**kw)
        self.dims = (dim,) + interdims
        self.vocsize = vocsize

        self.layers = []
        for i in range(len(self.dims)-1):
            layer = Forward(self.dims[i], self.dims[i+1], dropout=dropout)
            self.layers.append(layer)
        self.smo = SMO(self.dims[-1], outdim=self.vocsize)

    def apply(self, x):
        for layer in self.layers:
            x = layer(x)
        ret = self.smo(x)
        return ret


class StateToScalar(Block):
    def __init__(self, dim, outdim, **kw):
        super(StateToScalar, self).__init__(**kw)
        self.block = Forward(dim, outdim)
        self.agg = param((outdim,), name="scalartostate_agg").uniform()

    def apply(self, x):
        y = T.dot(x, self.block)
        z = T.dot(y, self.agg)      # (batsize,)
        ret = T.nnet.sigmoid(z)
        return ret


def make_vector_slicers(*sizes):
    sizes = list(sizes)
    alldim = sizes[0]
    sizes[0] = 0
    boundaries = [sizes[0]]
    del sizes[0]
    while len(sizes) > 0:
        boundaries.append(sizes[0]+boundaries[-1])
        del sizes[0]
    rets = []
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i + 1]
        yield Slicer(a, b)
        #yield MatMulSlicer(a, b, alldim)


class Slicer(Block):
    def __init__(self, a, b, **kw):
        super(Slicer, self).__init__(**kw)
        self.a = a
        self.b = b

    def apply(self, x):
        attrs = [slice(None, None, None)] * x.ndim
        if self.b - self.a == 1:
            attrs[-1] = self.a      #slice(self.a, self.b, None)     # or self.a
        else:
            attrs[-1] = slice(self.a, self.b, None)
        ret = x[attrs]
        return ret


class MatMulSlicer(Block):
    def __init__(self, a, b, alldim, **kw):
        super(MatMulSlicer, self).__init__(**kw)
        slicemat = np.eye(b - a)
        pre = np.zeros((a, b - a))
        post = np.zeros((alldim - b, b - a))
        slicemat = np.concatenate([pre, slicemat, post], axis=0)
        self.mat = Val(slicemat)

    def apply(self, x):
        ret = T.dot(x, self.mat)
        return ret


class MatParamGen(Block):
    def __init__(self, a, b, **kw):
        super(MatParamGen, self).__init__(**kw)
        self.W = param((a, b)).glorotuniform()

    def apply(self, x):
        return self.W


if __name__ == "__main__":
    from teafacto.blocks.seq.rnn import RNNWithoutInput
    m = RNNWithoutInput(3, 2)
    out = m(5)
    print out.eval().shape
    print out.eval()
