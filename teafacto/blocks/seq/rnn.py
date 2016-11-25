from enum import Enum
import numpy as np

from teafacto.blocks.seq.attention import WeightedSumAttCon, Attention, AttentionConsumer
from teafacto.blocks.seq.rnu import GRU, ReccableBlock, RecurrentBlock, RNUBase
from teafacto.blocks.basic import IdxToOneHot, Softmax, MatDot, Eye, VectorEmbed, Linear
from teafacto.core.base import Block, tensorops as T, asblock
from teafacto.util import issequence, isnumber


class RecStack(ReccableBlock):
    # must handle RecurrentBlocks ==> can not recappl, if all ReccableBlocks ==> can do recappl
    # must give access to final states of internal layers
    # must give access to all outputs of top layer
    # must handle masks
    def __init__(self, *layers, **kw):
        super(RecStack, self).__init__(**kw)
        self.layers = layers

    @property
    def numstates(self):
        return reduce(lambda x, y: x + y, [x.numstates for x in self.layers if isinstance(x, RecurrentBlock)], 0)

    # FWD API. initial states can be set, mask is accepted, everything is returned. Works for all RecurrentBlocks
    # FWD API IMPLEMENTED USING FWD API
    def innerapply(self, seq, mask=None, initstates=None):
        states = []
        for layer in self.layers:
            if isinstance(layer, RecurrentBlock):
                if initstates is not None:
                    layerinpstates = initstates[:layer.numstates]
                    initstates = initstates[layer.numstates:]
                else:
                    layerinpstates = None
                final, seq, layerstates = layer.innerapply(seq, mask=mask, initstates=layerinpstates)
                states.extend(layerstates)
            elif isinstance(layer, Block):
                seq = self.recurnonreclayer(seq, layer)
                final = seq[:, -1, :]
            else:
                raise Exception("can not apply this layer: " + str(layer) + " in RecStack")
        return final, seq, states           # full history of final output and all states (ordered from bottom layer to top)

    @classmethod
    def apply_mask(cls, xseq, maskseq=None):
        if maskseq is None:
            ret = xseq
        else:
            mask = T.tensordot(maskseq, T.ones((xseq.shape[2],)), 0)  # f32^(batsize, seqlen, outdim) -- maskseq stacked
            ret = mask * xseq
        return ret

    @classmethod
    def dummyrec(cls, layer):
        def innerrec(x_t):
            return layer(x_t)

        return innerrec

    @classmethod
    def recurnonreclayer(cls, x, layer):
        y = T.scan(fn=cls.dummyrec(layer),
                      sequences=x.dimswap(1, 0),
                      outputs_info=None)
        return y.dimswap(1, 0)

    # REC API: only works with ReccableBlocks
    def get_init_info(self, initstates):
        recurrentlayers = list(filter(lambda x: isinstance(x, ReccableBlock), self.layers))
        assert (len(filter(lambda x: isinstance(x, RecurrentBlock) and not isinstance(x, ReccableBlock),
                           self.layers)) == 0)  # no non-reccable blocks allowed
        if issequence(initstates):  # fill up init state args so that layers for which no init state is specified get default arguments that lets them specify a default init state
                                    # if is a sequence, expecting a value, not batsize
            if len(initstates) < self.numstates:    # top layers are being given the given init states, bottoms make their own default
                initstates = [initstates.shape[0]] * (self.numstates - len(initstates)) + initstates
        else:   # expecting a batsize as initstate arg
            initstates = [initstates] * self.numstates
        init_infos = []
        for recurrentlayer in recurrentlayers:  # from bottom layers to top
            arg = initstates[:recurrentlayer.numstates]
            initstates = initstates[recurrentlayer.numstates:]
            initinfo = recurrentlayer.get_init_info(arg)
            init_infos.extend(initinfo)
        return init_infos

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


class BiRNU(RecurrentBlock): # TODO: optimizer can't process this
    def __init__(self, fwd=None, rew=None, **kw):
        super(BiRNU, self).__init__(**kw)
        assert(isinstance(fwd, RNUBase) and isinstance(rew, RNUBase))
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
        return self.fwd.numstates + self.rew.numstates

    def innerapply(self, seq, mask=None, initstates=None):
        initstatesfwd = initstates[:self.fwd.numstates] if initstates is not None else initstates
        initstates = initstates[self.fwd.numstates:] if initstates is not None else initstates
        assert(initstates is None or len(initstates) == self.rew.numstates)
        initstatesrew = initstates
        fwdfinal, fwdout, fwdstates = self.fwd.innerapply(seq, mask=mask, initstates=initstatesfwd)   # (batsize, seqlen, innerdim)
        rewfinal, rewout, rewstates = self.rew.innerapply(seq, mask=mask, initstates=initstatesrew) # TODO: reverse?
        # concatenate: fwdout, rewout: (batsize, seqlen, feats) ==> (batsize, seqlen, feats_fwd+feats_rew)
        finalout = T.concatenate([fwdfinal, rewfinal], axis=1)
        out = T.concatenate([fwdout, rewout.reverse(1)], axis=2)
        return finalout, out, fwdstates+rewstates



class MaskMode(Enum):
    NONE = 0
    AUTO = 1
    AUTO_FORCE = 2

class MaskSetMode(Enum):
    NONE = 0
    ZERO = 1
    MASKID = 2


class MaskConfig(object):
    def __init__(self, maskmode=MaskMode.NONE, maskid=0, maskset=MaskSetMode.NONE):
        self.maskmode = maskmode
        self.maskid = maskid
        self.maskset = maskset

    def option(self, o):
        if isinstance(o, MaskSetMode):
            self.maskset = o
        elif isinstance(o, MaskMode):
            self.maskmode = o
        elif isinstance(o, int):
            self.maskid = o
        else:
            raise NotImplementedError("unrecognized mask configuration option")


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


class SeqEncoder(AttentionConsumer, Block):
    '''
    Encodes a sequence of vectors into a vector, input dims and output dims specified by the RNU unit
    Returns multiple outputs, multiple states
    Builds for one output
    '''

    def __init__(self, embedder, *layers, **kw):
        super(SeqEncoder, self).__init__(**kw)
        self._returnings = {"enc"}
        self._nomask = False
        self._maskconfig = kw["maskcfg"] if "maskcfg" in kw else MaskConfig(MaskMode.AUTO, 0, MaskSetMode.NONE)
        #dropout = False if "dropout" not in kw else kw["dropout"]
        self.embedder = embedder
        if len(layers) > 0:
            if len(layers) == 1:
                self.block = layers[0]
                assert(isinstance(self.block, RecurrentBlock))
            else:
                self.block = RecStack(*layers)
        else:
            self.block = None

    def apply(self, seq, weights=None, mask=None): # seq: (batsize, seqlen, dim), weights: (batsize, seqlen) OR (batsize, seqlen, seqlen*, dim) ==> reduce the innermost seqlen
        mask = seq.mask if mask is None else mask
        # embed
        if self.embedder is None:
            seqemb = seq
        else:
            seqemb = self.embedder(seq)     # maybe this way of embedding is not so nice for memory
            mask = seqemb.mask if mask is None else mask
            # auto mask
            if self._maskconfig.maskmode == MaskMode.AUTO_FORCE or \
                    (mask is None and self._maskconfig.maskmode == MaskMode.AUTO) or \
                    mask == "auto":
                mask = self._autogenerate_mask(seq, seqemb)

        # full mask
        fullmask = None
        if mask is not None:
            fullmask = mask
        if weights is not None:
            fullmask = weights if fullmask is None else weights * fullmask
        final, outputs, states = self.block.innerapply(seqemb, mask=fullmask)
        if mask is not None:
            outputs.mask = mask
        return self._get_apply_outputs(final, outputs, states, mask)

    def _autogenerate_mask(self, seq, seqemb):
        assert(seqemb.ndim == 3)
        print "automasking in SeqEncoder (%s)" % __file__
        axes = range(2, seq.ndim)       # mask must be 2D
        if "int" in seq.dtype:       # ==> indexes  # mask must be 2D
            if seq.ndim < 2:
                raise AttributeError("CAN NOT GENERATE MASK FOR NON-SEQUENCE")
            elif seq.ndim == 2:
                seqformask = seq
            else:
                print "generating default mask for non-standard seq shape (SeqEncoder, %s)" % __file__
                seqformask = seq[(slice(None, None, None),) * 2 + (0,) * (seq.ndim-2)]
                #if self._maskconfig.maskid != 0:
                #    raise AttributeError("CAN NOT CREATE MASK USING CUSTOM MASKID %d BECAUSE OF NON-STANDARD SEQ (%d dims, %s)" % (self._maskconfig.maskid, seq.ndim, str(seq.dtype)))
                #mask = T.gt(seq.sum(axis=axes), 0)      # 0 is TERMINUS
            assert(seqformask.ndim == 2)
            mask = T.neq(seqformask, self._maskconfig.maskid)
        else:
            #TODO raise AttributeError("CAN NOT GENERATE MASK FOR NON-INT SEQ")
            mask = T.gt(seq.norm(2, axis=axes), 0)
        return mask

    def _get_apply_outputs(self, final, outputs, states, mask):
        ret = []
        if "enc" in self._returnings:       # final states of topmost layer
            ret.append(final)
        if "all" in self._returnings:       # states (over all time) of topmost layer
            rete = outputs       # (batsize, seqlen, dim) --> zero-fy according to mask
            if self._maskconfig.maskset == MaskSetMode.ZERO and mask is not None:
                fmask = T.tensordot(mask, T.ones((outputs.shape[2],)), 0)
                rete = rete * fmask
            rete.mask = mask
            ret.append(rete)
        if "states" in self._returnings:    # final states (over all layers)???
            pass # TODO: do we need to support this?
        if "mask" in self._returnings:
            ret.append(mask)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def _get_apply_outputs_old(self, outputs):
        output = outputs[0]
        if self._all_states:    # get the last values of each of the states
            states = self.block.get_states_from_outputs(outputs[1:])
            ret = [s[-1, :, :] for s in states]
        else:                   # get the last state of the final state
            ret = [output[:, -1, :]] #output is (batsize, innerdim)
        if self._alloutput:    # include stack output
            return ret + [output]
        else:
            if len(ret) == 1:   # if only topmost last state or only one stateful layer --> one output
                return ret[0]
            else:               # else: a list of outputs
                return ret

    ### FLUENT MASK SETTINGS
    def mask(self, maskcfg):
        assert(isinstance(maskcfg, MaskConfig))
        self._maskconfig = maskcfg
        return self

    def maskoptions(self, *opt):
        for o in opt:
            self.maskoption(o)
        return self

    def maskoption(self, maskoption):
        self._maskconfig.option(maskoption)
        return self

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
    #CNN = CNNSeqEncoder        # TODO

    @staticmethod
    def getemb(emb=None, embdim=None, vocsize=None, maskid=-1):
        if emb is False:
            assert(embdim is not None)
            return None, embdim
        elif emb is not None:
            return emb, emb.outdim
        else:
            if embdim is None:
                return IdxToOneHot(vocsize), vocsize
            else:
                return VectorEmbed(indim=vocsize, dim=embdim, maskid=maskid), embdim


class RNNSeqEncoder(SeqEncoder):
    def __init__(self, indim=500, inpembdim=100, inpemb=None,
                 innerdim=200, bidir=False, maskid=None,
                 dropout_in=False, dropout_h=False, **kw):
        self.bidir = bidir
        inpemb, inpembdim = SeqEncoder.getemb(inpemb, inpembdim, indim, maskid=maskid)
        if not issequence(innerdim):
            innerdim = [innerdim]
        #self.outdim = innerdim[-1] if not bidir else innerdim[-1] * 2
        layers, lastdim = MakeRNU.make(inpembdim, innerdim, bidir=bidir,
                                       dropout_in=dropout_in, dropout_h=dropout_h)
        self.outdim = lastdim
        super(RNNSeqEncoder, self).__init__(inpemb, *layers, **kw)


class SeqDecoderOld(Block):
    """ seq decoder with attention with new inconcat implementation """
    def __init__(self, layers, softmaxoutblock=None, innerdim=None,
                 attention=None, inconcat=True, outconcat=False, **kw):
        super(SeqDecoderOld, self).__init__(**kw)
        self.embedder = layers[0]
        self.block = RecStack(*layers[1:])
        self.outdim = innerdim
        self.attention = attention
        self.inconcat = inconcat
        self.outconcat = outconcat
        self._mask = False
        self._attention = None
        assert(isinstance(self.block, ReccableBlock))
        if softmaxoutblock is None: # default softmax out block
            sm = Softmax()
            self.lin = MatDot(indim=self.outdim, dim=self.embedder.indim)
            self.softmaxoutblock = asblock(lambda x: sm(self.lin(x)))
        elif softmaxoutblock is False:
            self.softmaxoutblock = asblock(lambda x: x)
        else:
            self.softmaxoutblock = softmaxoutblock

    @property
    def numstates(self):
        return self.block.numstates

    def _get_seq_emb_t0(self, num, startsymemb=None):
        # seq_emb = self.embedder(seq[:, 1:])    # (batsize, seqlen-1, embdim)
        dim = self.embedder.outdim
        seq_emb_t0_sym = T.zeros((dim,), dtype="float32") if startsymemb is None else startsymemb
        seq_emb_t0 = T.repeat(seq_emb_t0_sym[np.newaxis, :], num, axis=0)
        return seq_emb_t0

    def apply(self, context, seq, context_0=None, initstates=None, mask=None,
              encmask=None, startsymemb=None, **kw):  # context: (batsize, enc.innerdim), seq: idxs-(batsize, seqlen)
        if initstates is None:
            initstates = seq.shape[0]
        elif issequence(initstates):
            if len(initstates) < self.numstates:  # fill up with batsizes for lower layers
                initstates = [seq.shape[0]] * (self.numstates - len(initstates)) + initstates
        init_info, nonseq = self.get_init_info(context, initstates,
                                    ctx_0=context_0, encmask=encmask)  # sets init states to provided ones
        embedder = self.embedder
        def recemb(x):
            return embedder(x)
        seq_emb = T.scan(fn=recemb, sequences=seq[:, 1:].dimswap(1, 0))
        seq_emb = seq_emb.dimswap(1, 0)
        seq_emb_t0 = self._get_seq_emb_t0(seq_emb.shape[0], startsymemb=startsymemb)
        seq_emb = T.concatenate([seq_emb_t0.dimshuffle(0, "x", 1), seq_emb], axis=1)
        outputs = T.scan(fn=self.rec,
                            sequences=seq_emb.dimswap(1, 0),
                            outputs_info=[None] + init_info,
                            non_sequences=nonseq)
        ret = outputs[0].dimswap(1, 0)  # returns probabilities of symbols --> (batsize, seqlen, vocabsize)
        if mask == "auto":
            mask = (seq > 0).astype("int32")
        ret = self.applymask(ret, mask)
        return ret

    @classmethod
    def applymask(cls, xseq, maskseq):
        if maskseq is None:
            return xseq
        else:
            mask = T.tensordot(maskseq, T.ones((xseq.shape[2],)), 0)  # f32^(batsize, seqlen, outdim) -- maskseq stacked
            masker = T.concatenate(
                [T.ones((xseq.shape[0], xseq.shape[1], 1)),
                 T.zeros((xseq.shape[0], xseq.shape[1], xseq.shape[2] - 1))],
                axis=2)  # f32^(batsize, seqlen, outdim) -- gives 100% prob to output 0
            ret = xseq * mask + masker * (1.0 - mask)
            return ret

    def get_init_info(self, context, initstates, ctx_0=None, encmask=None):
        initstates = self.block.get_init_info(initstates)
        ctx_0 = self._get_ctx_t(context, initstates, encmask) if ctx_0 is None else ctx_0
        if encmask is None:
            encmask = T.ones(context.shape[:2], dtype="float32")
        return [ctx_0, 0] + initstates, [encmask, context]

    def _get_ctx_t(self, ctx, states_tm1, encmask):
        if ctx.d.ndim == 2:     # static context
            ctx_t = ctx
            assert(self.attention is None)
        elif ctx.d.ndim > 2:
            # ctx is 3D, always dynamic context
            assert(self.attention is not None)
            h_tm1 = states_tm1[0]   # ??? --> will it also work with multi-state RNUs?
            ctx_t = self.attention(h_tm1, ctx, mask=encmask)
        return ctx_t

    def rec(self, x_t_emb, ctx_tm1, t, *args):  # x_t_emb: (batsize, embdim), context: (batsize, enc.innerdim)
        states_tm1 = args[:-2]
        ctx = args[-1]
        encmask = args[-2]
        #x_t_emb = self.embedder(x_t)  # i_t: (batsize, embdim)
        # do inconcat
        i_t = T.concatenate([x_t_emb, ctx_tm1], axis=1) if self.inconcat else x_t_emb
        rnuret = self.block.rec(i_t, *states_tm1)
        t += 1
        h_t = rnuret[0]
        states_t = rnuret[1:]
        ctx_t = self._get_ctx_t(ctx, states_t, encmask)  # get context with attention
        _y_t = T.concatenate([h_t, ctx_t], axis=1) if self.outconcat else h_t
        y_t = self.softmaxoutblock(_y_t)
        return [y_t, ctx_t, t] + states_t


class SeqDecoder(Block):
    """ seq decoder with attention with new inconcat implementation """

    def __init__(self, layers, softmaxoutblock=None, innerdim=None,
                 attention=None, inconcat=True, outconcat=False, dropout=False, **kw):
        super(SeqDecoder, self).__init__(**kw)
        self.embedder = layers[0]
        self.block = RecStack(*layers[1:])
        self.outdim = innerdim
        self.attention = attention
        self.inconcat = inconcat
        self.outconcat = outconcat
        self._mask = False
        self._attention = None
        assert (isinstance(self.block, ReccableBlock))
        if softmaxoutblock is None:  # default softmax out block
            sm = Softmax()
            self.lin = Linear(indim=self.outdim, dim=self.embedder.indim, dropout=dropout)
            self.softmaxoutblock = asblock(lambda x: sm(self.lin(x)))
        elif softmaxoutblock is False:
            self.softmaxoutblock = asblock(lambda x: x)
        else:
            self.softmaxoutblock = softmaxoutblock

    @property
    def numstates(self):
        return self.block.numstates

    def apply(self, ctx, seq, initstates=None, mask=None, ctxmask=None, **kw):  # context: (batsize, enc.innerdim), seq: idxs-(batsize, seqlen)
        batsize = seq.dshape[0]
        init_info, nonseqs = self.get_inits(initstates, batsize, ctx, ctxmask)
        seq_emb = self.embedder(seq)    # (batsize, seqlen, embdim)
        mask = seq_emb.mask if mask is None else mask
        outputs = T.scan(fn=self.inner_rec,
                            sequences=seq_emb.dimswap(1, 0),
                            outputs_info=[None] + init_info,
                            non_sequences=nonseqs)
        ret = outputs[0].dimswap(1, 0)  # returns probabilities of symbols --> (batsize, seqlen, vocabsize)
        ret.mask = mask
        return ret

    def get_inits(self, initstates=None, batsize=None, ctx=None, ctxmask=None):
        if initstates is None:
            initstates = batsize
        elif issequence(initstates):
            if len(initstates) < self.numstates:  # fill up with batsizes for lower layers
                initstates = [batsize * (self.numstates - len(initstates))] + initstates

        ctxmask = ctx.mask if ctxmask is None else ctxmask
        ctxmask = T.ones(ctx.shape[:2], dtype="float32") if ctxmask is None else ctxmask
        nonseqs = [ctxmask, ctx]
        return self.get_init_info(initstates), nonseqs

    def get_init_info(self, initstates):
        initstates = self.block.get_init_info(initstates)
        return initstates

    def rec(self, x_t, *args):
        x_t_emb = self.embedder(x_t)
        return self.inner_rec(x_t_emb, *args)

    def inner_rec(self, x_t_emb, *args):  # x_t_emb: (batsize, embdim), context: (batsize, enc.innerdim)
        states_tm1 = args[:-2]
        ctx = args[-1]
        encmask = args[-2]
        # x_t_emb = self.embedder(x_t)  # i_t: (batsize, embdim)
        # compute current context
        ctx_t = self._get_ctx_t(ctx, states_tm1[-1], encmask)     # TODO: might not work with LSTM
        # do inconcat
        i_t = T.concatenate([x_t_emb, ctx_t], axis=1) if self.inconcat else x_t_emb
        rnuret = self.block.rec(i_t, *states_tm1)
        h_t = rnuret[0]
        states_t = rnuret[1:]
        _y_t = T.concatenate([h_t, ctx_t], axis=1) if self.outconcat else h_t
        y_t = self.softmaxoutblock(_y_t)
        return [y_t] + states_t

    def _get_ctx_t(self, ctx, h_tm1, encmask):
        # ctx is 3D, always dynamic context
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
                 softmaxoutblock=None, dropout=False, dropout_h=False, **kw):
        layers, lastdim = self.getlayers(emb, embdim, embsize, maskid,
                           ctxdim, innerdim, rnu, inconcat, dropout, dropout_h)
        lastdim = lastdim if outconcat is False else lastdim + ctxdim
        super(SimpleRNNSeqDecoder, self).__init__(layers, softmaxoutblock=softmaxoutblock,
                                                  innerdim=lastdim, attention=attention, inconcat=inconcat,
                                                  outconcat=outconcat, dropout=dropout, **kw)

    def getlayers(self, emb, embdim, embsize, maskid, ctxdim, innerdim, rnu,
                  inconcat, dropout, dropout_h):
        emb, embdim = SeqDecoder.getemb(emb, embdim, embsize, maskid)
        firstdecdim = embdim + ctxdim if inconcat else embdim
        layers, lastdim = MakeRNU.make(firstdecdim, innerdim, bidir=False, rnu=rnu,
                                       dropout_in=dropout, dropout_h=dropout_h)
        layers = [emb] + layers
        return layers, lastdim


class MakeRNU(object):
    ''' generates a list of RNU's'''
    @staticmethod
    def make(initdim, specs, rnu=GRU, bidir=False, dropout_in=False, dropout_h=False):
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
            if fspec["bidir"]:
                rnn = BiRNU.fromrnu(fspec["rnu"], dim=prevdim, innerdim=fspec["dim"],
                                    dropout_in=dropout_in, dropout_h=dropout_h)
                prevdim = fspec["dim"] * 2
            else:
                rnn = fspec["rnu"](dim=prevdim, innerdim=fspec["dim"],
                                   dropout_in=dropout_in, dropout_h=dropout_h)
                prevdim = fspec["dim"]
            rnns.append(rnn)
        return rnns, prevdim

    @staticmethod
    def fromdims(innerdim, rnu=GRU, dropout_in=False, dropout_h=False):
        assert(len(innerdim) >= 2)
        initdim = innerdim[0]
        otherdim = innerdim[1:]
        return MakeRNU.make(initdim, otherdim, rnu=rnu,
                            dropout_in=dropout_in, dropout_h=dropout_h)

