from teafacto.core.base import Block, param, tensorops as T
from teafacto.blocks.basic import Forward
from teafacto.blocks.seq import GRU
from teafacto.blocks.seq.rnn import MakeRNU, RecStack
from teafacto.util import issequence


class DGTN(Block):
    numacts = 3

    def __init__(self,
                 # core settings
                 reltensor=None,
                 nsteps=None,
                 entembdim=None,
                 relembdim=None,
                 actembdim=None,
                 # internal RNN settings
                 innerdim=None,
                 rnu=GRU,
                 dropout_in=False,
                 dropout_h=False,
                 zoneout=False,
                 # what to use to compute next state
                 resultsummary=True,
                 actionsummary=True,
                 relationsummary=True,
                 entitysummary=True,
                 # coupling with encoder and attention #TODO
                 encoder=None,
                 attention=None,
                 inconcat=None,
                 outconcat=None,
                 **kw):
        super(DGTN, self).__init__(**kw)
        self.reltensor = reltensor
        self.nsteps = nsteps
        self.numrels = reltensor.shape[0]
        self.numents = reltensor.shape[1]
        self.entemb = param((self.numents, entembdim), name="entity_embeddings").glorotuniform()
        self.relemb = param((self.numrels, relembdim), name="relation_embeddings").glorotuniform()
        self.actemb = param((self.numacts, actembdim), name="action_embeddings").glorotuniform()
        # encoder and attention
        self.encoder = encoder
        self.attention = attention
        self.inconcat = inconcat
        self.outconcat = outconcat
        # core rnn
        indim = 0
        if resultsummary:
            indim += entembdim
        if actionsummary:
            indim += actembdim
        if relationsummary:
            indim += relembdim
        if entitysummary:
            indim += entembdim
        if self.inconcat:
            indim += self.encoder.outdim
        innerdim = innerdim if issequence(innerdim) else [innerdim]
        layers, lastdim = MakeRNU.fromdims([indim] + innerdim,
                                           rnu=rnu,
                                           dropout_h=dropout_h,
                                           dropout_in=dropout_in,
                                           zoneout=zoneout,
                                           param_init_states=True)
        self.block = RecStack(*layers)
        self.toattentionblock = Forward(lastdim, self.encoder.outdim)
        # interface from rnn to exec
        interfacedim = entembdim + relembdim + actembdim
        if self.outconcat:
            lastdim += self.encoder.outdim
        self.interfaceblock = Forward(lastdim, interfacedim)

    def apply(self, x):
        inpenc = self.encoder(x)
        batsize = inpenc.shape[0]
        # TODO: init state transfer from encoder??
        init_state = None
        init_info = self._get_init_states(init_state, batsize)
        init_input = None
        init_pointers = None
        nonseqs = self.get_nonseqs(inpenc)
        outputs = T.scan(fn=self.inner_rec,
                         outputs_info=[None] + [init_input, init_pointers] + init_info,
                         non_sequences=list(nonseqs),
                         n_steps=self.nsteps)

    def inner_rec(self, x_t, p_tm1, *args):
        ctx = args[-2]
        ctxmask = args[-1]
        states_tm1 = args[:-2]
        h_tm1 = states_tm1[-1]
        # do attention
        to_encoder = self.toattentionblock(h_tm1)
        ctx_t = self._get_ctx_t(ctx, to_encoder, self.attention, ctxmask)
        i_t = T.concatenate([x_t, ctx_t])
        if self.outconcat:
            to_interface = T.concatenate([h_tm1, ctx_t], axis=1)
        else:
            to_interface = h_tm1
        # do interfacing
        if_t = self.interfaceblock(to_interface)    # (batsize, ifdim)
        to_actselect = if_t[:, 0:self.actembdim]
        to_entselect = if_t[:, self.actembdim:self.actembdim+self.entembdim]
        to_relselect = if_t[:, self.actembdim+self.entembdim:self.actembdim+self.entembdim+self.relembdim]
        # execute interfaces
        act_weights, act_summary = self._get_act(to_actselect)
        ent_weights, ent_summary = self._get_ent(to_entselect)
        rel_weights, rel_summary = self._get_rel(to_relselect)



    @property
    def numstates(self):    return self.block.numstates

    def get_init_info(self, initstates):    return self.block.get_init_info(initstates)

    def _get_init_states(self, initstates, batsize):
        if initstates is None:
            initstates = batsize
        elif issequence(initstates):
            if len(initstates) < self.numstates:
                initstates = [batsize] * (self.numstates - len(initstates)) + initstates
        return self.get_init_info(initstates)

    def get_nonseqs(self, inpenc):
        ctx = inpenc
        ctxmask = ctx.mask if ctx.mask is not None else T.ones(ctx.shape[:2], dtype="float32")
        return ctx, ctxmask