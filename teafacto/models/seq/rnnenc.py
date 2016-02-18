from teafacto.core.rnn import *
from teafacto.core.trainutil import *

class EncSM(SMBase, Predictor, Saveable, Normalizable, Profileable):

    def defmodel(self):
        pathidxs = T.imatrix("pathidxs")
        zidx = T.ivector("zidx") # rhs corruption only
        scores = self.definnermodel(pathidxs) # ? scores: float(batsize, vocabsize)
        probs = T.nnet.softmax(scores) # row-wise softmax, ? probs: float(batsize, vocabsize)
        return probs, zidx, [pathidxs, zidx]

    @property
    def ownparams(self):
        return []

    @property
    def depparams(self):
        return []

    def getsamplegen(self, trainX, labels):
        batsize = self.batsize

        def samplegen():
            nonzeroidx = sorted(np.random.randint(0, trainX.shape[0], size=(batsize,)).astype("int32"))
            trainXsample = trainX[nonzeroidx, :].astype("int32")
            labelsample = labels[nonzeroidx].astype("int32")
            return [trainXsample, labelsample]     # start + path, target, bad_target
        return samplegen

    def getpredictfunction(self):
        probs, gold, inps = self.defmodel()
        score = probs[T.arange(gold.shape[0]), gold]
        scoref = theano.function(inputs=[inps[0], inps[1]], outputs=score)
        def pref(path, o):
            args = [np.asarray(i).astype("int32") for i in [path, o]]
            return scoref(*args)
        return pref
