import numpy as np

from use.modelusers import RecPredictor


class Searcher(object):
    def __init__(self, model, strategy=GreedySearch(), *buildargs, **kw):
        super(Searcher, self).__init__(**kw)
        self.strategy = strategy
        self.model = model
        self.recpred = RecPredictor(model, *buildargs)


class SeqTransDecSearcher(Searcher):
    # responsible for generating recappl prediction function from recappl of decoder
    """ Default: greedy search strategy """
    def decode(self, inpseq):
        stop = False
        i = 0
        curout = np.zeros((inpseq.shape[0])).astype("int32")
        accprobs = np.ones((inpseq.shape[0]))
        outs = []
        while not stop:
            curinp = inpseq[:, i]
            curprobs = self.recpred.feed(curinp, curout)
            accprobs *= np.max(curprobs, axis=1)
            curout = np.argmax(curprobs, axis=1).astype("int32")
            outs.append(curout)
            i += 1
            stop = i == inpseq.shape[1]
        #print accprobs
        ret = np.stack(outs).T
        assert (ret.shape == inpseq.shape)
        return ret, accprobs


class SeqEncDecSearch(Searcher):
    pass


# STRATEGIES
class SearchStrategy(object):
    pass


class GreedySearch(SearchStrategy):
    pass


class BeamSearch(SearchStrategy):
    pass


class VarBeamSearch(BeamSearch):
    pass

