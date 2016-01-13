
class RankingMetric(object):
    def __init__(self, **kw):
        self.reset()

    def reset(self):
        self.acc = 0.
        self.div = 0.

    @property
    def name(self):
        raise NotImplementedError("use subclass")

    def __call__(self, row=None, ranking=None): # df grouped by inputs, ranks ranks all entities
        if row is None or ranking is None:
            return self.compute()
        else:
            self.accumulate(row, ranking)

    def accumulate(self, row, ranking):
        raise NotImplementedError("use subclass")

    def compute(self):
        raise NotImplementedError("use subclass")


class RecallAt(RankingMetric):
    def __init__(self, top, **kw):
        self.topn = top
        super(RecallAt, self).__init__(**kw)

    @property
    def name(self):
        return "Recall @ %d" % self.topn

    def accumulate(self, row, ranking):
        topnvals = map(lambda x: x[0], ranking[:self.topn])
        correct = row[2]
        recall = len(set(correct).intersection(set(topnvals))) * 1. / len(correct)
        self.acc += recall
        self.div += 1

    def compute(self):
        return self.acc / self.div


class MeanQuantile(RankingMetric):

    @property
    def name(self):
        return "Mean Quantile"

    def accumulate(self, row, ranking):
        correct = row[2]
        total = len(ranking)
        for correctone in correct:
            pos = 0
            for (x, y) in ranking:
                if x == correctone:
                    pos -= 1
                    break
                pos += 1
            self.acc += 1. * (total - pos) / total
            self.div += 1

    def compute(self):
        return self.acc / self.div
