from teafacto.core.base import tensorops as T, Block
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
    def __init__(self, lenc, renc, scorer=DotDistance(), **kw):
        self.l = lenc
        self.r = renc
        self.s = scorer
        super(MatchScore, self).__init__(**kw)

    def apply(self, left, right):
        return self.s(self.l(left), self.r(right))  # left: (batsize, dim), right: (batsize, dim)


class SeqMatchScore(MatchScore):
    def __init__(self, lenc, renc, aggregator=lambda x: T.sum(x, axis=1), **kw):
        self.agg = aggregator
        super(SeqMatchScore, self).__init__(lenc, renc, **kw)

    def apply(self, left, right):
        l = self.l(left)
        r = self.r(right)
        scores, _ = T.scan(self.rec, sequences=[l.dimswap(1, 0), r.dimswap(1, 0)])
        scores = scores.dimswap(1, 0)
        return self.agg(scores)

    def rec(self, left, right):
        return self.s(left, right)
