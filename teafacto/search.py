import numpy as np

from modelusers import RecPredictor


class Searcher(object):
    def __init__(self, model, beamsize=1, **kw):
        super(Searcher, self).__init__(**kw)
        self.beamsize = beamsize
        self.model = model
        self.mu = RecPredictor(model)


class SeqTransDecSearch(Searcher):
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
            curprobs = self.mu.feed(curinp, curout)
            accprobs *= np.max(curprobs, axis=1)
            curout = np.argmax(curprobs, axis=1).astype("int32")
            outs.append(curout)
            i += 1
            stop = i == inpseq.shape[1]
        #print accprobs
        ret = np.stack(outs).T
        assert (ret.shape == inpseq.shape)
        return ret, accprobs

    def decode2(self, inpseq):       # inpseq: idx^(batsize, seqlen)
        i = 0
        stop = False
        # prevpreds = [np.zeros((inpseq.shape[0], 1))]*self.beamsize
        acc = np.zeros((inpseq.shape[0], 1)).astype("int32")
        accprobs = np.ones((inpseq.shape[0]))
        while not stop:
            curinpseq = inpseq[:, :i+1]
            curprobs = self.model.predict(curinpseq, acc)   # curpred: f32^(batsize, prevpred.seqlen, numlabels)
            curpreds = np.argmax(curprobs, axis=2).astype("int32")
            accprobs = np.max(curprobs, axis=2)[:, -1] * accprobs
            acc = np.concatenate([acc, curpreds[:, -1:]], axis=1)
            i += 1
            stop = i == inpseq.shape[1]
        ret = acc[:, 1:]
        assert(ret.shape == inpseq.shape)
        return ret, accprobs