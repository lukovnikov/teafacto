from teafacto.core.base import Block, param, tensorops as T, Val
from teafacto.blocks.basic import Forward
from teafacto.blocks.activations import Softmax
from teafacto.blocks.seq import GRU
from teafacto.blocks.seq.rnn import MakeRNU, RecStack
from teafacto.util import issequence


class DGTN(Block):
    numacts = 6

    def __init__(self,
                 # core settings
                 reltensor=None,
                 nsteps=None,
                 entembdim=None,
                 relembdim=None,
                 actembdim=None,
                 # what to use as input to compute next state
                 pointersummary=True,
                 actionsummary=True,
                 relationsummary=True,
                 entitysummary=True,
                 **kw):
        super(DGTN, self).__init__(**kw)
        self.reltensor = Val(reltensor)
        self.nsteps = nsteps
        self.numrels = reltensor.shape[0]
        self.numents = reltensor.shape[1]
        self.actembdim = actembdim
        self.entembdim = entembdim
        self.relembdim = relembdim
        self.pointersummary = pointersummary
        self.actionsummary = actionsummary
        self.entitysummary = entitysummary
        self.relationsummary = relationsummary
        self.entemb = param((self.numents, self.entembdim), name="entity_embeddings").glorotuniform()
        self.relemb = param((self.numrels, self.relembdim), name="relation_embeddings").glorotuniform()
        self.actemb = param((self.numacts, self.actembdim), name="action_embeddings").glorotuniform()
        self.core = None

    def set_core(self, core):
        self.core = core
        self.core.attentiontransformer = Forward(self.core.lastdim, self.core.encoder.outdim)
        interfacedim = self.entembdim + self.relembdim + self.actembdim
        self.interfaceblock = Forward(self.core.outconcatdim, interfacedim)

    def get_indim(self):        # has test
        indim = 0
        if self.pointersummary:
            indim += self.entembdim*2
        if self.actionsummary:
            indim += self.actembdim
        if self.relationsummary:
            indim += self.relembdim
        if self.entitysummary:
            indim += self.entembdim
        return indim

    @property
    def encoder(self):
        return self.core.encoder

    def apply(self, x):     # TODO TEST
        inpenc = self.encoder(x)
        batsize = inpenc.shape[0]
        init_info, nonseqs = self.get_inits(batsize, inpenc)
        outputs = T.scan(fn=self.inner_rec,
                         outputs_info=[None] + init_info,
                         non_sequences=list(nonseqs),
                         n_steps=self.nsteps)
        lastmainpointer = outputs[0][-1, :, :]
        return lastmainpointer

    def get_inits(self, batsize, ctx):      # TODO TEST
        init_info, nonseqs = self.core.get_inits(batsize, ctx)
        x_0 = T.zeros((batsize, self.get_indim()))
        p_0 = T.zeros((batsize, 2, self.numents))
        return [x_0, p_0] + init_info, nonseqs

    def inner_rec(self, x_t, p_tm1, *args):     # TODO TEST
        p_tm1_main = p_tm1[:, 0, :]
        p_tm1_aux = p_tm1[:, 1, :]
        # execute rnn
        rnuret = self.core.inner_rec(x_t, *args)
        y_t = rnuret[0]
        newargs = rnuret[1:]
        # get interfaces
        if_t = self.interfaceblock(y_t)    # (batsize, ifdim)
        to_actselect = if_t[:, 0:self.actembdim]
        to_entselect = if_t[:, self.actembdim:self.actembdim+self.entembdim]
        to_relselect = if_t[:, self.actembdim+self.entembdim:self.actembdim+self.entembdim+self.relembdim]
        # compute interface weights
        act_weights = self._get_att(to_actselect, self.actemb)
        ent_weights = self._get_att(to_entselect, self.entemb)
        rel_weights = self._get_att(to_relselect, self.relemb)
        # execute ops
        p_t_main, p_t_aux = T.zeros_like(p_tm1_main), T.zeros_like(p_tm1_aux)
        p_t_main, p_t_aux = \
            self._merge_exec(self._exec_preserve(p_tm1_main, p_tm1_aux), p_t_main, p_t_aux, act_weights[:, 0])
        p_t_main, p_t_aux = \
            self._merge_exec(self._exec_find(ent_weights, p_tm1_main, p_tm1_aux), p_t_main, p_t_aux, act_weights[:, 1])
        p_t_main, p_t_aux = \
            self._merge_exec(self._exec_hop(rel_weights, p_tm1_main, p_tm1_aux), p_t_main, p_t_aux, act_weights[:, 2])
        p_t_main, p_t_aux = \
            self._merge_exec(self._exec_intersect(p_tm1_main, p_tm1_aux), p_t_main, p_t_aux, act_weights[:, 3])
        p_t_main, p_t_aux = \
            self._merge_exec(self._exec_union(p_tm1_main, p_tm1_aux), p_t_main, p_t_aux, act_weights[:, 4])
        p_t_main, p_t_aux = \
            self._merge_exec(self._exec_difference(p_tm1_main, p_tm1_aux), p_t_main, p_t_aux, act_weights[:, 5])
        p_t_main, p_t_aux = \
            self._merge_exec(self._exec_swap(p_tm1_main, p_tm1_aux), p_t_main, p_t_aux, act_weights[:, 6])

        p_t = T.concatenate([p_t_main, p_t_aux], axis=1)

        # summarize pointers and weights
        act_summ = self._summarize_by_prob(act_weights, self.actemb)
        ent_summ = self._summarize_by_prob(ent_weights, self.entemb)
        rel_summ = self._summarize_by_prob(rel_weights, self.relemb)
        p_main_summ = self._summarize_by_pointer(p_t_main, self.entemb)
        p_aux_summ = self._summarize_by_pointer(p_t_aux, self.entemb)
        # compute next input
        concatdis = []
        if self.pointersummary:     concatdis += [p_main_summ, p_aux_summ]
        if self.actionsummary:      concatdis += [act_summ]
        if self.entitysummary:      concatdis += [ent_summ]
        if self.relationsummary:    concatdis += [rel_summ]
        x_tp1 = T.concatenate(concatdis, axis=1)
        o_t = p_t_main
        return [o_t, x_tp1, p_t] + newargs

    # EXECUTION METHODS     # all below have tests
    def _merge_exec(self, newps, oldmain, oldaux, w):
        w = w.dimadd(1)
        newmain, newaux = newps
        oldmain += newmain * w
        oldaux += newaux * w
        return oldmain, oldaux

    def _exec_preserve(self, p_tm1_main, p_tm1_aux):
        return p_tm1_main, p_tm1_aux

    def _exec_find(self, w, oldmain, oldaux):
        return w, oldmain

    def _exec_hop(self, relw, oldmain, oldaux):  # (batsize, nument), (batsize, numrel)
        relmats = T.tensordot(relw, self.reltensor, axes=([1], [0]))   # (batsize, nument, nument)
        newp = T.batched_dot(oldmain, relmats)
        newp = T.clip(newp, 0, 1)   # prevent overflow
        return newp, oldaux

    def _exec_intersect(self, oldmain, oldaux):     # prod or min?
        return oldmain * oldaux, T.zeros_like(oldaux)

    def _exec_union(self, a, b):    # max or clipped sum?
        return T.maximum(a, b), T.zeros_like(b)

    def _exec_difference(self, a, b):
        return T.clip(a - b, 0, 1), T.zeros_like(b)

    def _exec_swap(self, a, b):
        return b, a

    # HELPER METHODS        # have tests
    def _get_att(self, crit, data): # (batsize, critdim), (num, embdim) -> (batsize, num)
        att = T.tensordot(crit, data, axes=([1], [1]))
        att = Softmax()(att)
        return att

    def _summarize_by_prob(self, w, data):  # (batsize, num), (num, embdim)
        ret = T.tensordot(w, data, axes=([1], [0]))     # (batsize, embdim)
        return ret

    def _summarize_by_pointer(self, w, data):   # (batsize, num), (num, embdim)
        ret = T.tensordot(w, data, axes=([1], [0]))     # (batsize, embdim)
        nor = T.sum(w, axis=1, keepdims=True)      # (batsize,)
        ret = ret / (nor + 1e-6)            # normalize to average
        return ret


    @property
    def numstates(self):    return self.core.numstates
