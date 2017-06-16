from teafacto.core.base import Block, param, tensorops as T, Val
from teafacto.blocks.basic import Forward
from teafacto.blocks.activations import Softmax, GumbelSoftmax, MaxHot
from teafacto.blocks.seq import GRU
from teafacto.blocks.seq.rnn import MakeRNU, RecStack
from teafacto.util import issequence
from teafacto.blocks.loss import Loss, CrossEntropy, BinaryCrossEntropy


# TODO: write tests (esp. for with attention)

class DGTN(Block):
    numacts = 7
    _max_in_hop = True
    _min_in_intersect = False

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
                 gumbel=False,
                 maxhot=False,
                 **kw):
        super(DGTN, self).__init__(**kw)
        self.reltensor = Val(reltensor.astype("float32"))
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
        self._encoder = None
        self._gumbel_sm = gumbel
        self._maxhot_sm = maxhot

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
        if self._encoder is not None:
            return self._encoder
        elif hasattr(self.core, "encoder"):
            return self.core.encoder
        else:
            return None

    @encoder.setter
    def encoder(self, enc):
        self._encoder = enc

    def apply(self, x, p_main_0=None):
        inpenc = self.encoder(x)        # 2D or 3D
        batsize = inpenc.shape[0]
        init_info, nonseqs = self.get_inits(batsize, inpenc, p_main_0=p_main_0)
        outputs = T.scan(fn=self.inner_rec,
                         outputs_info=[None] + init_info,
                         non_sequences=list(nonseqs),
                         n_steps=self.nsteps)
        lastmainpointer = outputs[0][-1, :, :]
        return lastmainpointer

    def get_inits(self, batsize, ctx, p_main_0=None):
        init_info, nonseqs = self.core.get_inits(batsize, ctx)
        x_0 = T.zeros((batsize, self.get_indim()))
        if p_main_0 is not None:
            p_0 = T.concatenate([p_main_0.dimadd(1), T.zeros((batsize, 1, self.numents))], axis=1)
        else:
            p_0 = T.zeros((batsize, 2, self.numents))
        return [x_0, p_0] + init_info, nonseqs

    def inner_rec(self, x_t, p_tm1, *args):
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
        ent_weights = self._get_att(to_entselect, self.entemb, override_custom_sm=True)
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

        p_t = T.concatenate([p_t_main.dimadd(1), p_t_aux.dimadd(1)], axis=1)

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
        if self._max_in_hop:
            x = oldmain.dimadd(2)
            newp = T.max(x * relmats, axis=1)
        else:
            newp = T.batched_dot(oldmain, relmats)
            newp = T.clip(newp, 0, 1)   # prevent overflow
        return newp, oldaux

    def _exec_intersect(self, oldmain, oldaux):     # prod or min?
        if self._min_in_intersect:
            return T.minimum(oldmain, oldaux), T.zeros_like(oldaux)
        else:
            return oldmain * oldaux, T.zeros_like(oldaux)

    def _exec_union(self, a, b):    # max or clipped sum?
        return T.maximum(a, b), T.zeros_like(b)

    def _exec_difference(self, a, b):
        return T.clip(a - b, 0, 1), T.zeros_like(b)

    def _exec_swap(self, a, b):
        return b, a

    # HELPER METHODS        # have tests
    def _get_att(self, crit, data, override_custom_sm=False): # (batsize, critdim), (num, embdim) -> (batsize, num)
        att = T.tensordot(crit, data, axes=([1], [1]))
        if self._gumbel_sm and not override_custom_sm:
            sm = GumbelSoftmax(deterministic_pred=True)
        elif self._maxhot_sm and not override_custom_sm:
            sm = MaxHot(ste=True)
        else:
            sm = Softmax()
        att = sm(att)
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


class KLPointerLoss(Loss):        # does softmax on pointer, then KL div
    def __init__(self, lrg=1e0, softmaxnorm=True, **kw):
        super(KLPointerLoss, self).__init__(**kw)
        self.LRG = lrg
        self.softmaxnorm = softmaxnorm

    def apply(self, pred, gold):        # (batsize, numents)
        EPS = 1e-6
        if self.softmaxnorm:
            pred_sm = Softmax()((pred-1)*self.LRG)
        else:
            pred_sm = pred / T.sum(pred, axis=1, keepdims=True) + EPS
        gold_sm = gold / T.sum(gold, axis=1, keepdims=True) + EPS
        cross_entropy = CrossEntropy()(pred_sm, gold_sm)
        gold_entropy = CrossEntropy()(gold_sm, gold_sm)
        kl_div = cross_entropy - gold_entropy
        return kl_div


class PWPointerLoss(Loss):
    def __init__(self, balanced=False, **kw):
        self.balanced = balanced
        super(PWPointerLoss, self).__init__(**kw)

    def apply(self, pred, gold):
        bces = BinaryCrossEntropy(sum_per_example=False)(pred, gold)
        if self.balanced:
            posces = gold * bces
            negces = (1 - gold) * bces
            posctr = T.sum(posces, axis=1) / T.sum(gold, axis=1)
            negctr = T.sum(negces, axis=1) / T.sum(1-gold, axis=1)
            ret = 0.5 * posctr + 0.5 * negctr
        else:
            ret = T.sum(bces, axis=1)
        return ret




