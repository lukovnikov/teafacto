import numpy as np
from enum import Enum

from teafacto.blocks.basic import IdxToOneHot, Softmax, MatDot, VectorEmbed, Linear, Forward, SMO
from teafacto.blocks.seq.attention import AttentionConsumer
from teafacto.blocks.seq.rnu import GRU, ReccableBlock, RecurrentBlock, RNUBase, ReccableWrapper
from teafacto.core.base import Block, tensorops as T, asblock
from teafacto.util import issequence
from teafacto.blocks.activations import *


class RecStack(ReccableBlock):
    # must handle RecurrentBlocks ==> can not recappl, if all ReccableBlocks ==> can do recappl
    # must give access to final states of internal layers
    # must give access to all outputs of top layer
    # must handle masks
    def __init__(self, *layers, **kw):
        super(RecStack, self).__init__(**kw)
        self.layers = []
        for l in layers:
            if isinstance(l, RecurrentBlock):
                self.layers.append(l)
            elif isinstance(l, Block):
                self.layers.append(ReccableWrapper(l))
            else:
                raise Exception("cannot apply this layer")

    @property
    def numstates(self):
        return reduce(lambda x, y: x + y, [x.numstates for x in self.layers if isinstance(x, RecurrentBlock)], 0)

    def get_statespec(self, flat=False):
        ret = tuple([l.get_statespec(flat=flat) for l in self.layers])
        if flat:
            ret = reduce(lambda x, y: x + y, ret, tuple())
        return ret

    @property
    def outdim(self):
        return self.layers[-1].outdim

    # FWD API. initial states can be set, mask is accepted, everything is returned. Works for all RecurrentBlocks
    # FWD API IMPLEMENTED USING FWD API
    def innerapply(self, seq, mask=None, initstates=None):
        states = []     # bottom states first
        seqs = [seq]
        for layer in self.layers:
            if initstates is not None:
                layerinpstates = initstates[:layer.numstates]
                initstates = initstates[layer.numstates:]
            else:
                layerinpstates = None
            finals, seqs, layerstates = layer.innerapply(*seqs, mask=mask, initstates=layerinpstates)
            states.extend(layerstates)
        return finals, seqs, states           # full history of final output and all states (ordered from bottom layer to top)

    @classmethod
    def apply_mask(cls, xseq, maskseq=None):
        if maskseq is None:
            ret = xseq
        else:
            mask = T.tensordot(maskseq, T.ones((xseq.shape[2],)), 0)  # f32^(batsize, seqlen, outdim) -- maskseq stacked
            ret = mask * xseq
        return ret

    # REC API: only works with ReccableBlocks
    def get_init_info(self, initstates):
        recurrentlayers = list(filter(lambda x: isinstance(x, ReccableBlock), self.layers))
        assert (len(filter(lambda x: isinstance(x, RecurrentBlock) and not isinstance(x, ReccableBlock),
                           self.layers)) == 0)  # no non-reccable blocks allowed
        if issequence(initstates):  # fill up init state args so that layers for which no init state is specified get default arguments that lets them specify a default init state
                                    # if is a sequence, expecting a value, not batsize
            if len(initstates) < self.numstates:    # top layers are being given the given init states, bottoms make their own default
                initstates = [None] * (self.numstates - len(initstates)) + initstates
            batsize = 0
            for initstate in initstates:
                if initstate is not None:
                    batsize = initstate.shape[0]
            initstates = [batsize if initstate is None else initstate for initstate in initstates]
        else:   # expecting a batsize as initstate arg
            initstates = [initstates] * self.numstates
        init_infos = []
        for recurrentlayer in recurrentlayers:  # from bottom layers to top
            arg = initstates[:recurrentlayer.numstates]
            initstates = initstates[recurrentlayer.numstates:]
            initinfo = recurrentlayer.get_init_info(arg)
            init_infos = init_infos + initinfo
        return init_infos       # left is bottom

    def rec(self, x_t, *states):
        # apply each block on x_t to get next-level input, consume states in the process
        nextinp = x_t
        nextstates = []
        for block in self.layers:
            if isinstance(block, ReccableBlock):
                numstates = block.numstates
                recstates = states[:numstates]
                states = states[numstates:]
                rnuret = block.rec(nextinp, *recstates)
                nextstates.extend(rnuret[1:])
                nextinp = rnuret[0]
            elif isinstance(block, Block): # block is a function
                nextinp = block(nextinp)
        return [nextinp] + nextstates


class FluentSeqEncoderBuilder(object):
    """ Fluent construction interface for seqencoder """
    def __init__(self, **kw):
        super(FluentSeqEncoderBuilder, self).__init__(**kw)
        self.embedder = None
        self.layers = []
        self.lastdim = None

    def _return(self):
        return self

    def make(self):
        if self.layers is not None and len(self.layers) > 0:
            return SeqEncoder(self.embedder, *self.layers)
        else:
            raise Exception("not ready")

    def noembedder(self, dim):
        self.lastdim = dim
        return self._return()

    def onehotembedder(self, vocsize):
        self.embedder, self.lastdim = IdxToOneHot(vocsize), vocsize
        return self._return()

    def setembedder(self, block):
        self.embedder = block
        self.lastdim = self.embedder.outdim
        return self._return()

    def vectorembedder(self, vocsize, embdim, maskid=None):
        self.embedder, self.lastdim = VectorEmbed(indim=vocsize, dim=embdim, maskid=maskid), embdim
        return self._return()

    def setlayers(self, *layers):
        self.layers = layers
        return self._return()

    def addlayers(self, dim=None, bidir=False, dropout_in=False, dropout_h=False, zoneout=False, rnu=GRU, **kw):
        inpdim = self.lastdim
        if not issequence(dim):
            dim = [dim]
        #self.outdim = innerdim[-1] if not bidir else innerdim[-1] * 2
        layers, lastdim = MakeRNU.make(inpdim, dim, bidir=bidir, rnu=rnu,
                                       dropout_in=dropout_in, dropout_h=dropout_h, zoneout=zoneout)
        self.layers = self.layers + layers
        self.lastdim = lastdim
        return self._return()

    def add_layer(self, layer, lastdim=None):
        self.layers.append(layer)
        self.lastdim = lastdim
        return self._return()

    def add_forward_layers(self, dim=None, activation=Tanh, dropout=False, nobias=True):
        inpdim = self.lastdim
        if not issequence(dim):
            dim = [dim]
        dim = [inpdim] + dim
        for d1, d2 in zip(dim[:-1], dim[1:]):
            self.layers += [Forward(d1, d2, activation=activation(), dropout=dropout, nobias=nobias)]
            self.lastdim = d2
        return self._return()

    @staticmethod
    def getemb(emb=None, embdim=None, vocsize=None, maskid=None):
        if emb is False:
            assert (embdim is not None)
            return None, embdim
        elif emb is not None:
            return emb, emb.outdim
        else:
            if embdim is None:
                return IdxToOneHot(vocsize), vocsize
            else:
                return VectorEmbed(indim=vocsize, dim=embdim, maskid=maskid), embdim


class SeqEncoder(AttentionConsumer, Block):
    '''
    Encodes a sequence of vectors into a vector, input dims and output dims specified by the RNU unit
    Returns multiple outputs, multiple states
    Builds for one output
    '''

    def __init__(self, embedder, *layers, **kw):
        super(SeqEncoder, self).__init__(**kw)
        self._returnings = {"enc"}
        self.embedder = embedder
        if len(layers) > 0:
            self.block = RecStack(*layers)
        else:
            self.block = None

    def get_statespec(self, flat=False):
        return self.block.get_statespec(flat=flat)

    @property
    def outdim(self):
        return self.block.outdim

    def apply_argspec(self):
        return ((2, "int"),) if self.embedder is not None else ((3, "float"),)

    def apply(self, seq, weights=None): # seq: (batsize, seqlen, dim), weights: (batsize, seqlen) OR (batsize, seqlen, seqlen*, dim) ==> reduce the innermost seqlen
        # embed
        if self.embedder is None:
            seqemb = seq
        else:
            seqemb = self.embedder(seq)     # maybe this way of embedding is not so nice for memory
        mask = seqemb.mask
        # full mask
        fullmask = None
        if mask is not None:
            fullmask = mask
        if weights is not None:
            fullmask = weights if fullmask is None else weights * fullmask
        #embed()
        finals, outputs, states = self.block.innerapply(seqemb, mask=fullmask)   # bottom states first
        for output in outputs:
            output.mask = mask
        return self._get_apply_outputs(finals, outputs, states, mask)

    def _get_apply_outputs(self, finals, outputs, states, mask):
        ret = []
        if "enc" in self._returnings:       # final states of topmost layer
            if len(finals) == 1:
                finals = finals[0]
            ret.append(finals)
        if "all" in self._returnings:       # states (over all time) of topmost layer
            for output in outputs:
                output.mask = mask
            if len(outputs) == 1:
                outputs = outputs[0]
            # (batsize, seqlen, dim) --> zero-fy according to mask
            ret.append(outputs)
        if "mask" in self._returnings:
            ret.append(mask)
            pass
        if "states" in self._returnings:    # final states (over all layers)???
            ret.append(states)      #pass # TODO: do we need to support this
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    ### FLUENT OUTPUT SETTINGS
    def reset_return(self):
        self._returnings.clear()
        return self

    def with_states(self):       # TODO
        '''Call this switch to get the final states of all recurrent layers'''
        self._returnings.add("states")
        return self

    def all_outputs(self):
        '''Call this switch to get the actual output of top layer as the last outputs'''
        self.reset_return()
        self._returnings.add("all")
        return self

    def with_outputs(self):
        self._returnings.add("all")
        return self

    def with_mask(self):
        ''' Calling this switch also returns the mask on original idx input sequence'''
        self._returnings.add("mask")
        return self

    def setreturn(self, *args):
        self.reset_return()
        for arg in args:
            self._returnings.add(arg)
        return self

    @staticmethod
    def RNN(*args, **kwargs):
        return RNNSeqEncoder(*args, **kwargs)

    @staticmethod
    def simple(*args, **kwargs):
        return RNNSeqEncoder(*args, **kwargs)

    @staticmethod
    def fluent():
        return FluentSeqEncoderBuilder()

    @staticmethod
    def getemb(emb=None, embdim=None, vocsize=None, maskid=-1):
        return FluentSeqEncoderBuilder.getemb(emb=emb, embdim=embdim, vocsize=vocsize, maskid=maskid)


class RNNSeqEncoder(SeqEncoder):
    def __init__(self, indim=500, inpembdim=100, inpemb=None,
                 innerdim=200, bidir=False, maskid=None,
                 dropout_in=False, dropout_h=False, rnu=GRU, **kw):
        self.bidir = bidir
        inpemb, inpembdim = SeqEncoder.getemb(inpemb, inpembdim, indim, maskid=maskid)
        if not issequence(innerdim):
            innerdim = [innerdim]
        #self.outdim = innerdim[-1] if not bidir else innerdim[-1] * 2
        layers, lastdim = MakeRNU.make(inpembdim, innerdim, bidir=bidir, rnu=rnu,
                                       dropout_in=dropout_in, dropout_h=dropout_h)
        self._lastdim = lastdim
        super(RNNSeqEncoder, self).__init__(inpemb, *layers, **kw)

    @property
    def outdim(self):
        return self._lastdim

    @staticmethod
    def fluent():
        return FluentSeqEncoderBuilder()


class RiboRNN(Block):
    numcontrols = 2     # input shift and output shift
    # TODO: can also be trained softly, without ste-maxhot

    def __init__(self,
                 inpemb=None,
                 innerdim=None, rnu=GRU,
                 dropout_in=False, dropout_h=False, zoneout=False,
                 smo=None,
                 nsteps=None, **kw):
        super(RiboRNN, self).__init__(**kw)
        self.inpemb = inpemb
        indim = inpemb.outdim
        self.nsteps = nsteps
        self.smo = smo
        innerdim = innerdim if issequence(innerdim) else [innerdim]
        self.outdim = innerdim[-1]
        self.outvocsize = self.smo.outdim
        #innerdim[-1] += 1
        # make the layers
        layers, lastdim = MakeRNU.fromdims([indim] + innerdim,
                                           rnu=rnu,
                                           dropout_h=dropout_h,
                                           dropout_in=dropout_in,
                                           zoneout=zoneout,
                                           )
        self.rnublock = RecStack(*layers)
        self.control_block = Forward(lastdim, self.numcontrols, activation=Sigmoid())

        self.position_softmax_temp = 0.5
        self.maxhot_position_during_prediction = True
        self.maxhot_position_during_training = False
        self.init_mem_to_zero_index = True

        self._debug_memory_evolution = False

    def apply(self, seq):   # (batsize, seqlen) - int^vocsize
        batsize = seq.shape[0]
        seq_emb = self.inpemb(seq)
        mask = seq_emb.mask     # (batsize, seqlen, embdim) ~ float
        init_states = self.rnublock.get_init_info(batsize)  # init rnu's from zero
        # initialize output memory, slides
        if self.init_mem_to_zero_index:
            EPS = 1e-9
            outmem_0 = T.concatenate([T.ones((batsize, self.nsteps, 1)) - EPS * (self.outvocsize - 1),
                                       T.zeros((batsize, self.nsteps, self.outvocsize - 1)) + EPS],
                                      axis=-1)
        else:   # init to zeros
            outmem_0 = T.zeros((batsize, self.nsteps, self.outvocsize))
        control_acc_0 = T.zeros((batsize, self.numcontrols))
        timestep = T.zeros((batsize,))
        outputs = T.scan(fn=self.inner_rec,
                         sequences=None,
                         outputs_info=[None, outmem_0, control_acc_0, timestep] + init_states,
                         non_sequences=seq_emb,
                         n_steps=self.nsteps)
        ret = outputs[1][-1]      # last state of outmem
        ret.mask = mask
        if not self._debug_memory_evolution:
            return ret     # RUN
        else:
            return outputs[1]   # DEBUG

    def inner_rec(self, outmem_tm1, control_acc_tm1, timestep, *states_tm1):
        x = states_tm1[-1]      # (batsize, seqlen, inpembdim)
        states_tm1 = states_tm1[:-1]    # !!! top state may have few extra elems
        x_t = self._get_x_t(x, control_acc_tm1)
        o_t, states_t = self.rnublock.rec(x_t, *states_tm1)
        y_t = self._get_y_t(o_t)
        control_t = self._get_control(states_t, o_t, y_t)    # (batsize, numcontrols)
        control_acc_t = self._update_control(control_acc_tm1, control_t)
        outmem_t, write_pos = self._update_outmem(y_t, outmem_tm1, control_acc_tm1) # (batsize, outseqlen, outvocsize)
        timestep += 1
        return write_pos, outmem_t, control_acc_t, timestep, states_t

    # used by rec api, can deviate
    def _get_y_t(self, o_t):
        return self.smo(o_t)

    # used by rec api, can deviate
    def _get_control(self, states_t, o_t, y_t):
        # given states, output vector and output symbol probs, get control vector
        ret = self.control_block(o_t)   # (batsize, numcontrols)
        shift = Threshold(0.0, ste=True)(ret[:, 0:1] - 0.5)
        ret = T.concatenate([shift, shift], axis=-1)    # same shift for input and output
        return ret      # (batsize, numcontrols)

    ### keep standard below
    # is simulated outside rec api, keep standard and simple
    def _get_x_t(self, x, control_acc_tm1):  # (batsize, seqlen, inpembdim) and (batsize, ctrldim)
        # given input over all timesteps and control vectors, get current input
        crit = control_acc_tm1[:, 0]    # (batsize,)
        # 1. get address in the input   (attention weights)
        pos = self.__get_position_weights(crit, x.shape[1], maskout="future")
        # 2. get summary of x using the addressing
        pos = pos.dimadd(2)
        summary = T.sum(x * pos, axis=1)
        return summary  # (batsize, inpembdim)

    # is simulated outside rec api, keep standard and simple
    def _update_outmem(self, y_t, outmem_tm1,
                       control_acc_tm1):  # (batsize, outvocsize) and (batsize, maxoutseqlen, outvocsize) and (batsize, numcontrols)
        crit = control_acc_tm1[:, 1]    # (batsize,)
        # 1. get address in the output (attention weights) based on control
        pos = self.__get_position_weights(crit, outmem_tm1.shape[1], maskout="past")
        # 2. update outmem with y_t based on the addressing
        pos = pos.dimadd(2) # (batsize, maxoutseqlen, 1)
        y = y_t.dimadd(1)   # (batsize, 1, outvocsize)
        outmem_t = outmem_tm1 * (1 - pos) + pos * y
        return outmem_t, pos

    def __get_position_weights(self, crit, maxlen, maskout="future"):
        # TODO: apply softmax-maxhot(-ste) during training, maxhot during prediction
        # if softmax during training, probably masking out the past/future is necessary
        #       in order not to affect past/future predictions
        dist = self.__get_raw_position_distances(crit, maxlen)
        timemask = self.__get_position_time_mask(crit, maxlen, mode=maskout)
        dist.mask = timemask
        weights = Softmax(temperature=self.position_softmax_temp,
                          maxhot=self.maxhot_position_during_training,     # default False
                          maxhot_ste=True,
                          maxhot_pred=self.maxhot_position_during_prediction)\
            (dist)
        return weights  # (batsize, seqlen)

    def __get_raw_position_distances(self, crit, maxlen):    # (batsize,)
        # max hot
        positions = T.arange(0, maxlen, dtype="float32")  # (maxlen,)
        positions = positions.dimadd(0)
        crit = crit.dimadd(1)
        dist = abs(positions - crit) ** 2   # (batsize, maxlen)
        return -dist

    def __get_position_time_mask(self, crit, maxlen, mode="future"):
        positions = T.arange(0, maxlen)  # (maxlen,)
        positions = positions.dimadd(0)
        crit = crit.dimadd(1)
        eps = 0.5
        if mode == "future":
            mask = Threshold(0.0)(crit - positions + eps)   # all positions strictly larger than crit will be zero
        elif mode == "past":
            mask = Threshold(0.0)(positions - crit + eps)   # all positions strictly smaller than crit will be zero
        else:
            raise Exception("mode not recognized")
        return mask

    # is simulated outside rec api, keep standard and simple
    def _update_control(self, control_acc_tm1, control_t):  # don't override this
        return control_acc_tm1 + control_t

    def rec(self, x_t, *args):
        """ takes current input, outputs output and shift control scalars"""
        pass        # TODO


class SeqDecoder(Block):

    def __init__(self, layers, softmaxoutblock=None, innerdim=None,
                 attention=None,
                 inconcat=True, outconcat=False, stateconcat=True, updatefirst=False,
                 dropout=False, **kw):
        super(SeqDecoder, self).__init__(**kw)
        self.embedder = layers[0]
        self.block = RecStack(*layers[1:])
        self.outdim = innerdim
        self.attention = attention
        self.inconcat = inconcat
        self.outconcat = outconcat
        self.stateconcat = stateconcat
        assert(self.stateconcat or self.outconcat)
        self.updatefirst = updatefirst
        self._mask = False
        self._attention = None
        assert (isinstance(self.block, ReccableBlock))
        if softmaxoutblock is None:  # default softmax out block
            self.softmaxoutblock = SMO(self.outdim, self.embedder.indim, dropout=dropout)
        elif softmaxoutblock is False:
            self.softmaxoutblock = asblock(lambda x: x)
        else:
            self.softmaxoutblock = softmaxoutblock

    @property
    def numstates(self):
        return self.block.numstates

    def get_statespec(self, flat=False):
        return self.block.get_statespec(flat=flat)

    def apply_argspec(self):
        return ((3, "float"), (2, "int"))

    def apply(self, ctx, seq, initstates=None, **kw):  # context: (batsize, enc.innerdim), seq: idxs-(batsize, seqlen)
        batsize = seq.shape[0]
        init_info, nonseqs = self.get_inits(initstates, batsize, ctx)
        seq_emb = self.embedder(seq)    # (batsize, seqlen, embdim)
        mask = seq_emb.mask
        outputs = T.scan(fn=self.inner_rec,
                            sequences=seq_emb.dimswap(1, 0),
                            outputs_info=[None] + init_info,
                            non_sequences=nonseqs)
        ret = outputs[0].dimswap(1, 0)  # returns probabilities of symbols --> (batsize, seqlen, vocabsize)
        ret.mask = mask
        return ret

    def get_inits(self, initstates=None, batsize=None, ctx=None):
        if initstates is None:
            initstates = batsize
        elif issequence(initstates):
            if len(initstates) < self.numstates:  # fill up with batsizes for lower layers
                initstates = [batsize * (self.numstates - len(initstates))] + initstates

        ctxmask = ctx.mask
        ctxmask = T.ones(ctx.shape[:(ctx.ndim-1)], dtype="float32") if ctxmask is None else ctxmask
        nonseqs = [ctxmask, ctx]

        ctx_0 = T.zeros((batsize, ctx.shape[2] if self.attention is not None else ctx.shape[1]))
        return [ctx_0] + self.get_init_info(initstates), nonseqs

    def get_init_info(self, initstates):
        initstates = self.block.get_init_info(initstates)
        return initstates

    def rec(self, x_t, *args):
        x_t_emb = self.embedder(x_t)
        return self.inner_rec(x_t_emb, *args)

    def inner_rec(self, x_t_emb, ctx_tm1, *args):  # x_t_emb: (batsize, embdim)
        states_tm1 = args[:-2]
        ctx = args[-1]                    # (batsize, inseqlen, inencdim)
        encmask = args[-2]
        # x_t_emb = self.embedder(x_t)  # i_t: (batsize, embdim)
        # compute current context
        y_tm1 = states_tm1[-1]      # left is bottom, left is inner --> should work for lstm
        if self.updatefirst:
            i_t = T.concatenate([x_t_emb, ctx_tm1], axis=1) if self.inconcat else x_t_emb
            rnuret = self.block.rec(i_t, *states_tm1)
            o_t = rnuret[0]
            states_t = rnuret[1:]
            y_t = states_t[-1]
            ctx_t = self._get_ctx_t(ctx, y_t, encmask)
        else:
            ctx_t = self._get_ctx_t(ctx, y_tm1, encmask)
            i_t = T.concatenate([x_t_emb, ctx_t], axis=1) if self.inconcat else x_t_emb
            rnuret = self.block.rec(i_t, *states_tm1)
            o_t = rnuret[0]
            states_t = rnuret[1:]
        # output
        concatthis = []
        if self.stateconcat:           concatthis.append(o_t)
        if self.outconcat:             concatthis.append(ctx_t)
        _y_t = T.concatenate(concatthis, axis=1) if len(concatthis) > 1 else concatthis[0]
        y_t = self.softmaxoutblock(_y_t)
        return [y_t, ctx_t] + states_t

    def _get_ctx_t(self, ctx, h_tm1, encmask):
        if self.attention is not None:
            assert(ctx.d.ndim > 2)
            ctx_t = self.attention(h_tm1, ctx, mask=encmask)
            return ctx_t
        else:
            return ctx

    @staticmethod
    def getemb(*args, **kwargs):
        return SeqEncoder.getemb(*args, **kwargs)

    @staticmethod
    def RNN(*args, **kwargs):
        return SimpleRNNSeqDecoder(*args, **kwargs)


class SimpleRNNSeqDecoder(SeqDecoder):
    def __init__(self, emb=None, embdim=None, embsize=None, maskid=-1,
                 ctxdim=None, innerdim=None, rnu=GRU,
                 inconcat=True, outconcat=False, attention=None,
                 softmaxoutblock=None, dropout=False, dropout_h=False, zoneout=False, **kw):
        layers, lastdim = self.getlayers(emb, embdim, embsize, maskid,
                           ctxdim, innerdim, rnu, inconcat,
                                         dropout, dropout_h, zoneout)
        lastdim = lastdim if outconcat is False else lastdim + ctxdim
        super(SimpleRNNSeqDecoder, self).__init__(layers, softmaxoutblock=softmaxoutblock,
                                                  innerdim=lastdim, attention=attention, inconcat=inconcat,
                                                  outconcat=outconcat, dropout=dropout, **kw)

    def getlayers(self, emb, embdim, embsize, maskid, ctxdim, innerdim, rnu,
                  inconcat, dropout, dropout_h, zoneout):
        emb, embdim = SeqDecoder.getemb(emb, embdim, embsize, maskid)
        firstdecdim = embdim + ctxdim if inconcat else embdim
        layers, lastdim = MakeRNU.make(firstdecdim, innerdim, bidir=False, rnu=rnu,
                                       dropout_in=dropout, dropout_h=dropout_h)
        layers = [emb] + layers
        return layers, lastdim


class MakeRNU(object):
    ''' generates a list of RNU's'''
    @staticmethod
    def make(initdim, specs, rnu=GRU, bidir=False,
             dropout_in=False, dropout_h=False, zoneout=False,
             param_init_states=False):
        if not issequence(specs):
            specs = [specs]
        rnns = []
        prevdim = initdim
        for spec in specs:
            fspec = {"dim": None, "bidir": bidir, "rnu": rnu}
            if isinstance(spec, int):
                fspec["dim"] = spec
            elif isinstance(spec, dict):
                assert(hasattr(spec, "dim")
                       and
                       set(spec.keys()).union(set(fspec.keys()))
                            == set(fspec.keys()))
                fspec.update(spec)
            noinput = prevdim == None
            if fspec["bidir"]:
                assert(not noinput)
                rnn = BiRNU.fromrnu(fspec["rnu"], dim=prevdim, innerdim=fspec["dim"],
                                    dropout_in=dropout_in, dropout_h=dropout_h, zoneout=zoneout,
                                    param_init_states=param_init_states)
                prevdim = fspec["dim"] * 2
            else:
                rnn = fspec["rnu"](dim=prevdim, innerdim=fspec["dim"], noinput=noinput,
                                   dropout_in=dropout_in, dropout_h=dropout_h, zoneout=zoneout,
                                   param_init_states=param_init_states)
                prevdim = fspec["dim"]
            rnns.append(rnn)
        return rnns, prevdim

    @staticmethod
    def fromdims(innerdim, rnu=GRU, bidir=False,
                 dropout_in=False, dropout_h=False, zoneout=False,
                 param_init_states=False):
        assert(len(innerdim) >= 2)
        initdim = innerdim[0]
        otherdim = innerdim[1:]
        return MakeRNU.make(initdim, otherdim, rnu=rnu, bidir=bidir,
                            dropout_in=dropout_in, dropout_h=dropout_h, zoneout=zoneout,
                            param_init_states=param_init_states)


class BiRNU(RecurrentBlock): # TODO: optimizer can't process this
    def __init__(self, fwd=None, rew=None, **kw):
        super(BiRNU, self).__init__(**kw)
        assert(isinstance(fwd, RNUBase) and isinstance(rew, RNUBase))
        assert(type(fwd) == type(rew))
        self.fwd = fwd
        self.rew = rew
        assert(self.fwd.indim == self.rew.indim)

    @classmethod
    def fromrnu(cls, rnucls, *args, **kw):
        assert(issubclass(rnucls, RNUBase))
        kw["reverse"] = False
        fwd = rnucls(*args, **kw)
        kw["reverse"] = True
        rew = rnucls(*args, **kw)
        return cls(fwd=fwd, rew=rew)

    @property
    def numstates(self):
        assert(self.fwd.numstates == self.rew.numstates)
        return self.fwd.numstates

    def get_statespec(self, flat=False):
        ret = []
        fwdspec = self.fwd.get_statespec(flat=flat)
        rewspec = self.rew.get_statespec(flat=flat)
        assert(len(fwdspec) == len(rewspec))
        for fwdspece, rewspece in zip(fwdspec, rewspec):
            assert(fwdspece[0] == rewspece[0])
            ret.append((fwdspece[0], (fwdspece[1][0] + rewspece[1][0],)))
        return ret

    def innerapply(self, seq, mask=None, initstates=None):
        initstatesfwd = initstates[:self.fwd.numstates] if initstates is not None else initstates
        initstates = initstates[self.fwd.numstates:] if initstates is not None else initstates
        assert(initstates is None or len(initstates) == self.rew.numstates)
        initstatesrew = initstates
        fwdfinals, fwdouts, fwdstates = self.fwd.innerapply(seq, mask=mask, initstates=initstatesfwd)   # (batsize, seqlen, innerdim)
        rewfinals, rewouts, rewstates = self.rew.innerapply(seq, mask=mask, initstates=initstatesrew) # TODO: reverse?
        # concatenate: fwdout, rewout: (batsize, seqlen, feats) ==> (batsize, seqlen, feats_fwd+feats_rew)
        finalouts = [T.concatenate([fwdfinal, rewfinal], axis=1) for fwdfinal, rewfinal in zip(fwdfinals, rewfinals)]
        outs = [T.concatenate([fwdout, rewout.reverse(1)], axis=2) for fwdout, rewout in zip(fwdouts, rewouts)]
        states = []
        for fwdstate, rewstate in zip(fwdstates, rewstates):
            states.append(T.concatenate([fwdstate, rewstate], axis=2))      # for taking both final states, we need not reverse
        return finalouts, outs, states


class EncLastDim(Block):
    def __init__(self, enc, **kw):
        super(EncLastDim, self).__init__(**kw)
        self.enc = enc

    def apply(self, x, mask=None):
        if self.enc.embedder is None:
            mindim = 3
            maskdim = x.ndim - 1
        else:
            mindim = 2
            maskdim = x.ndim
        if mask is not None:
            assert(mask.ndim == maskdim)
        else:
            mask = T.ones(x.shape[:maskdim])
        if x.ndim == mindim:
            return self.enc(x, mask=mask)
        elif x.ndim > mindim:
            ret = T.scan(fn=self.outerrec, sequences=[x, mask], outputs_info=None)
            return ret
        else:
            raise Exception("cannot have less than {} dims".format(mindim))

    def outerrec(self, xred, mask):  # x: ndim-1
        ret = self.apply(xred, mask=mask)
        return ret

    @property
    def outdim(self):
        return self.enc.outdim


class RNNWithoutInput(Block):
    def __init__(self, dims=None, layers=None, dropout=False, **kw):
        super(RNNWithoutInput, self).__init__(**kw)
        if issequence(dims):
            assert(layers is None)
            self.dims = dims
        else:
            if layers is None:
                layers = 1
            assert(dims is not None)
            self.dims = [dims] * layers
        self.layers, _ = MakeRNU.make(None, self.dims, bidir=False,
                                   param_init_states=True,
                                   dropout_h=dropout,
                                   dropout_in=dropout)
        self.block = RecStack(*self.layers)

    def apply(self, steps):
        initinfo = self.block.get_init_info(2)
        seqs = T.zeros((steps, 2, 2))
        outputs = T.scan(self.block.rec, sequences=seqs,
                         outputs_info=[None] + initinfo)
        return outputs[0][:, 0, :]

