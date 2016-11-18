from teafacto.core.base import tensorops as T, Block, asblock, param
from teafacto.blocks.basic import Linear
from teafacto.util import issequence
from IPython import embed

#region ======== SCORES =============

#region ######## DISTANCES ########

class Distance(Block):
    def apply(self, l, r):
        raise NotImplementedError("use subclass")


class DotDistance(Block):
    def apply(self, l, r):  # l: f32^(batsize, dim), r: f32^(batsize, dim)
        return T.batched_dot(r, l)


class CosineDistance(Block):
    def apply(self, l, r):  # l: f32^(batsize, dim), r:f32^(batsize, dim)
        dots = T.batched_dot(r, l)
        lnorms = l.norm(2, axis=-1)
        rnorms = r.norm(2, axis=-1)
        while lnorms.ndim < dots.ndim:
            lnorms = T.shape_padaxis(lnorms, -1)
        while rnorms.ndim < dots.ndim:
            rnorms = T.shape_padaxis(rnorms, -1)
        ret = dots/(lnorms*rnorms + 1e-6)
        ret.push_extra_outs({"lnorms": lnorms, "rnorms": rnorms})
        return ret


class EuclideanDistance(Block):
    def apply(self, l, r):
        return (l-r).norm(2, axis=1)


class LinearDistance(Block):
    def __init__(self, ldim, rdim, aggdim, **kw):
        super(LinearDistance, self).__init__(**kw)
        self.lin = Linear(indim=ldim, dim=aggdim)
        self.lin2 = Linear(indim=rdim, dim=aggdim)
        self.agg = param((aggdim,), name="attention_agg").uniform()

    def apply(self, l, r):      # (batsize, dim)
        a = self.lin(l)     # (batsize, dim)
        b = self.lin2(r)    # (batsize, dim) or (batsize, seqlen, dim)
        x, s = a, b
        if a.ndim != b.ndim:
            x, s = (a, b) if a.ndim > b.ndim else (b, a)
            while s.ndim < x.ndim:
                s = T.shape_padaxis(s, 1)
        att = s + x
        att = self._gateit(att)
        ret = T.dot(att, self.agg)  # (batsize,)
        return ret

    def _gateit(self, x):
        return x


class LinearGateDistance(LinearDistance):
    def __init__(self, *args, **kw):
        activation = T.tanh
        if "activation" in kw:
            activation = kw["activation"]
            del kw["activation"]
        self.activation = activation
        super(LinearGateDistance, self).__init__(*args, **kw)

    def _gateit(self, x):
        return self.activation(x)


class BilinearDistance(Block):
    def __init__(self, ldim, rdim, **kw):
        super(BilinearDistance, self).__init__(**kw)
        self.W = param((rdim, ldim), name="gendotdist").glorotuniform()

    def apply(self, l, r):  # (batsize, dims)
        ldot = T.dot(r, self.W) # (batsize, rdim)
        ret = T.batched_dot(ldot, l)  # (batsize, )
        return ret

#endregion

#endregion


class MatchScore(Block):
    def __init__(self, lenc, renc, scorer=DotDistance(),
                 argproc=lambda x, y: ((x,), (y,)), **kw):
        self.l = lenc
        self.r = renc
        self.s = scorer
        self.argproc = argproc
        super(MatchScore, self).__init__(**kw)

    def apply(self, *args):
        left, right = self.argproc(*args)
        l = self.l(*left)
        r = self.r(*right)
        return self.innerapply(l, r)

    def innerapply(self, l, r):
        return self.s(l, r)


class SeqMatchScore(MatchScore):
    def __init__(self, lenc, renc,
                 aggregator=asblock(lambda x: T.sum(x, axis=1)), **kw):
        self.agg = aggregator
        super(SeqMatchScore, self).__init__(lenc, renc, **kw)

    def innerapply(self, l, r):
        scores = T.scan(self.rec, sequences=[l.dimswap(1, 0), r.dimswap(1, 0)])
        scores = scores.dimswap(1, 0)
        ret = self.agg(scores)
        print ret.ndim
        return ret

    def rec(self, left, right):
        return self.s(left, right)
