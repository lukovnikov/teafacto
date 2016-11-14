import numpy as np

from teafacto.use.modelusers import RecPredictor
from teafacto.util import isnumber


# OLD --> remove
class Searcher(object):
    def __init__(self, model, stopsymbol=None, **kw):
        super(Searcher, self).__init__(**kw)
        self.model = model
        self.stopsymbol = stopsymbol        # TODO use it
        self.recpred = RecPredictor(model)

    def init(self, *args):
        self.recpred.init(*args)
        return self

    def search(self, *args, **kwargs):
        raise NotImplementedError("use subclass")


# OLD
class SeqTransDecSearcher(Searcher):
    # responsible for generating recappl prediction function from recappl of decoder
    """ Default: greedy search strategy """
    def search(self, inpseq):
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


# MODEL WRAPPERS
class ModelWrapper(object):
    def __init__(self, model, startsymbol=0, stopsymbol=None, **kw):
        super(ModelWrapper, self).__init__(**kw)
        self.args = []
        self.recpred = RecPredictor(model)
        self.startsymbol = startsymbol
        self.stopsymbol = stopsymbol

    def init(self, *args):
        self.recpred.init(*args)
        return self

    @staticmethod
    def wrap(model, startsymbol=0, stopsymbol=None):
        assert(startsymbol is not None and isnumber(startsymbol))
        if hasattr(model, "searchwrapper"):
            return model.searchwrapper(model, startsymbol=startsymbol, stopsymbol=stopsymbol)
        else:
            raise Exception("no search wrapper assigned to model")

    def _get_cur_probs(self, i, curout):
        if self.stopsymbol is not None:
            stopmask = curout == self.stopsymbol
            probs = self.get_cur_probs(i, curout)
            probs[stopmask, :] = 0.
            probs[stopmask, self.stopsymbol] = 1.
            return probs
        else:
            return self._get_cur_probs(i, curout)

    # can override this:
    def setargs(self, *args):
        self.args = args

    # implement this:
    def get_cur_probs(self, i, curout):
        raise NotImplementedError("use subclass")


class SeqTransDecWrapper(ModelWrapper):
    def setargs(self, *args):
        assert(len(args), 1)
        self.args = args[0]

    def init_out(self):
        return self.startsymbol * np.ones((self.args.shape[0],)).astype("int32")

    def get_cur_probs(self, i, curout):
        curinp = self.args[:, i]
        curprobs = self.recpred.feed(curinp, curout)
        return curprobs

    def isstop(self, i):
        return i == self.args.shape[1]


# STRATEGIES
class SearchStrategy(object):
    def __init__(self, model, startsymbol=0, stopsymbol=None, maxlen=100, **kw):
        assert(startsymbol is not None)
        super(SearchStrategy, self).__init__(**kw)
        self.wrapped = ModelWrapper.wrap(model,
                     startsymbol=startsymbol,
                     stopsymbol=stopsymbol)
        self.stopsymbol = stopsymbol        # TODO use it
        self.maxlen = maxlen

    def init(self, *args):
        self.wrapped.init(*args)
        return self


class GreedySearch(SearchStrategy):
    def search(self, *args):
        self.wrapped.setargs(*args)
        stop = False
        i = 0
        curout = self.wrapped.init_out()
        accprobs = None
        outs = []
        while not stop:
            curprobs = self.wrapped._get_cur_probs(i, curout)
            if accprobs is None:
                accprobs = np.ones((curprobs.shape[0],))       # batsize
            accprobs *= np.max(curprobs, axis=1)
            curout = np.argmax(curprobs, axis=1).astype("int32")
            outs.append(curout)
            i += 1
            stop = (i == (self.maxlen - 1)) \
                   or self.wrapped.isstop(i) \
                   or np.all(curout == self.stopsymbol)

        ret = np.stack(outs).T
        return ret, accprobs


class BeamSearch(SearchStrategy):
    def __init__(self, model, beamsize=10, startsymbol=0, stopsymbol=None, maxlen=100, **kw):
        super(BeamSearch, self).__init__(model, startsymbol=startsymbol, stopsymbol=stopsymbol, maxlen=maxlen, **kw)
        self.beamsize = beamsize

    def search(self, *args):
        self.wrapped.setargs(*args)
        # TODO


class VarBeamSearch(BeamSearch):
    pass

