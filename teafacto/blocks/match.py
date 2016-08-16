from teafacto.core.base import tensorops as T, Block, asblock
from teafacto.util import issequence
from IPython import embed

#region ======== SCORES =============

#region ######## DISTANCES ########
class DotDistance(Block):
    def apply(self, l, r):  # l: f32^(batsize, dim), r: f32^(batsize, dim)
        return T.batched_dot(l, r)


class CosineDistance(Block):
    def apply(self, l, r):  # l: f32^(batsize, dim), r:f32^(batsize, dim)
        dots = T.batched_dot(l, r)
        lnorms = l.norm(2, axis=1)
        rnorms = r.norm(2, axis=1)
        return dots/(lnorms*rnorms + 1e-6)


class EuclideanDistance(Block):
    def apply(self, l, r):
        return (l-r).norm(2, axis=1)

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
        scores, _ = T.scan(self.rec, sequences=[l.dimswap(1, 0), r.dimswap(1, 0)])
        scores = scores.dimswap(1, 0)
        return self.agg(scores)

    def rec(self, left, right):
        return self.s(left, right)
