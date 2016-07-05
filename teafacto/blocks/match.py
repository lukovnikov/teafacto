from teafacto.core.base import tensorops as T, Block


class MatchScore(Block):
    def __init__(self, lenc, renc, scorer=T.batched_dot, **kw):
        self.l = lenc
        self.r = renc
        self.s = scorer
        super(MatchScore, self).__init__(**kw)

    def apply(self, left, right):
        return self.s(self.l(left), self.r(right))  # left: (batsize, dim), right: (batsize, dim)