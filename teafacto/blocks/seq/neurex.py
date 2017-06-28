from teafacto.core.base import Block, param, tensorops as T, Val, asblock
from teafacto.blocks.basic import Forward
from teafacto.blocks.activations import Softmax, GumbelSoftmax, MaxHot
from teafacto.blocks.seq import GRU
from teafacto.blocks.seq.rnn import MakeRNU, RecStack
from teafacto.util import issequence
from teafacto.blocks.loss import Loss, CrossEntropy, BinaryCrossEntropy
import numpy as np


# TODO: write tests (esp. for with attention)

class DGTN(Block):
    numacts = 7
    _max_in_hop = True
    _min_in_intersect = False
    EPS = 1e-6
    _act_name_to_int = {"_nop": 0, "find": 1, "hop": 2, "intersect": 3, "union": 4, "difference": 5, "swap": 6}

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
                 # override action sequence by fixed global action sequence
                 action_override=None,
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
        self._act_sel_mask = np.ones((self.numacts,))
        # return options
        self._ret_actions = False
        self._ret_entities = False
        self._ret_relations = False
        self._ret_all_main_ptrs = False
        # action override
        self._action_override = Val(action_override) if action_override is not None else None
        # temperatures
        self._act_temp = self._ent_temp = self._rel_temp = 1.

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
        if self._action_override is None:
            action_override = T.zeros((self.nsteps, batsize, self.numacts))
        else:
            action_override = self._action_override.dimadd(1)\
                .repeat(batsize, axis=1)
        outputs = T.scan(fn=self.inner_rec,
                         sequences=[action_override],
                         outputs_info=[None]*4 + init_info,
                         non_sequences=list(nonseqs),
                         n_steps=self.nsteps)
        mainpointers = outputs[0].dimswap(0, 1)     # (batsize, nsteps, nentities)
        action_weights = outputs[1].dimswap(0, 1)   # (batsize, nsteps, nactions)
        entity_weights = outputs[2].dimswap(0, 1)   # (batsize, nsteps, nentities)
        relation_weights = outputs[3].dimswap(0, 1)
        lastmainpointer = mainpointers[:, -1, :]
        lastmainpointer = T.clip(lastmainpointer, 0+self.EPS, 1-self.EPS)
        return self._get_output(lastmainpointer, mainpointers,
                                action_weights, entity_weights, relation_weights)

    def _get_output(self, lmptr, mptr, aw, ew, rw):
        ret = (lmptr,)
        if self._ret_all_main_ptrs:
            ret += (mptr,)
        if self._ret_actions:
            ret += (aw,)
        if self._ret_entities:
            ret += (ew,)
        if self._ret_relations:
            ret += (rw,)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def _no_extra_ret(self):
        self._ret_all_main_ptrs = \
            self._ret_actions = \
            self._ret_entities = \
            self._ret_relations = False

    def get_inits(self, batsize, ctx, p_main_0=None):
        init_info, nonseqs = self.core.get_inits(batsize, ctx)
        x_0 = T.zeros((batsize, self.get_indim()))
        if p_main_0 is not None:
            p_0 = T.concatenate([p_main_0.dimadd(1), T.zeros((batsize, 1, self.numents))], axis=1)
        else:
            p_0 = T.zeros((batsize, 2, self.numents))
        return [x_0, p_0] + init_info, nonseqs

    def _get_interface(self, y_t):
        if_t = self.interfaceblock(y_t)    # (batsize, ifdim)
        to_actselect = if_t[:, 0:self.actembdim]
        to_entselect = if_t[:, self.actembdim:self.actembdim+self.entembdim]
        to_relselect = if_t[:, self.actembdim+self.entembdim:self.actembdim+self.entembdim+self.relembdim]
        return to_actselect, to_entselect, to_relselect

    def inner_rec(self, action_override_t, x_t, p_tm1, *args):
        p_tm1_main = p_tm1[:, 0, :]
        p_tm1_aux = p_tm1[:, 1, :]
        # execute rnn
        rnuret = self.core.inner_rec(x_t, *args)
        y_t = rnuret[0]
        newargs = rnuret[1:]
        # get interfaces
        to_actselect, to_entselect, to_relselect = self._get_interface(y_t)
        # compute interface weights
        if self._action_override is None:
            act_weights = self._get_att(to_actselect, self.actemb, mask=Val(self._act_sel_mask), temp=self._act_temp)
        else:
            act_weights = action_override_t
        ent_weights = self._get_att(to_entselect, self.entemb, temp=self._ent_temp)
        rel_weights = self._get_att(to_relselect, self.relemb, temp=self._rel_temp)
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
        return [o_t, act_weights, ent_weights, rel_weights, x_tp1, p_t] + newargs

    # ACTION METHODS     # all below have tests
    def disable(self, actionname):
        if actionname == "all":
            self._act_sel_mask[1:] = 0
        else:
            actaddr = self._act_name_to_int[actionname]
            self._act_sel_mask[actaddr] = 0
        return self

    def enable(self, actionname):
        actaddr = self._act_name_to_int[actionname]
        self._act_sel_mask[actaddr] = 1
        return self

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
    def _get_att(self, crit, data, mask=None, override_custom_sm=False, temp=1.): # (batsize, critdim), (num, embdim) -> (batsize, num)
        att = T.tensordot(crit, data, axes=([1], [1]))  # (batsize, num*)
        if mask is not None:    # (num*)
            mask = mask.dimadd(0).repeat(att.shape[0], axis=0)
            att.mask = mask
            print "masked attention"
        if self._gumbel_sm and not override_custom_sm:
            sm = GumbelSoftmax(deterministic_pred=True, temperature=temp)
        elif self._maxhot_sm and not override_custom_sm:
            sm = Softmax(maxhot=True, maxhot_ste=True, maxhot_pred=True, temperature=temp)
        else:
            sm = Softmax(maxhot_pred=True, temperature=temp)
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


class DGTN_S(DGTN):
    def __init__(self,
                 # core settings
                 reltensor=None,
                 nsteps=None,
                 entembdim=None,
                 actembdim=None,
                 attentiondim=None,
                 # what to use as input to compute next state
                 pointersummary=True,
                 actionsummary=True,
                 relationsummary=True,
                 entitysummary=True,
                 gumbel=False,
                 maxhot=False,
                 **kw):
        super(DGTN_S, self).__init__(reltensor=reltensor, nsteps=nsteps, entembdim=entembdim,
                                     relembdim=entembdim, actembdim=actembdim,
                                     pointersummary=pointersummary, actionsummary=actionsummary, entitysummary=entitysummary, relationsummary=relationsummary,
                                     gumbel=gumbel, maxhot=maxhot, **kw)
        self.attentiondim = attentiondim

    def set_core(self, core):
        self.core = core
        self.core.attentiontransformer = \
            asblock(lambda x: x[:, self.actembdim:self.actembdim+self.attentiondim])

    def _get_interface(self, y_t):
        to_actselect = y_t[:, 0:self.actembdim]
        to_entselect = y_t[:, -self.entembdim:]
        to_relselect = y_t[:, -self.entembdim:]
        return to_actselect, to_entselect, to_relselect


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


class PointerRecall(Loss):
    EPS = 1e-6

    def apply(self, pred, gold):
        pred = pred > 0.5   # (batsize, numents), 0 or 1 (same for gold)
        tp = T.sum(pred * gold, axis=1)
        recall_norm = T.sum(gold, axis=1) + self.EPS
        recall = tp / recall_norm
        return recall


class PointerPrecision(Loss):
    EPS = 1e-6

    def apply(self, pred, gold):
        pred = pred > 0.5
        tp = T.sum(pred * gold, axis=1)
        prec_norm = T.sum(pred, axis=1) + self.EPS
        precision = tp / prec_norm
        return precision


class PointerFscore(Loss):
    EPS = 1e-6

    def apply(self, pred, gold):
        recall = PointerRecall()(pred, gold)
        precision = PointerPrecision()(pred, gold)
        fscore = 2 * recall * precision / (recall + precision + self.EPS)
        return fscore





