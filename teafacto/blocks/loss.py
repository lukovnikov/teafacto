from teafacto.core import T, Block
from lasagne.objectives import *


class Loss(Block):
    def __init__(self, **kwargs):
        super(Loss, self).__init__(**kwargs)

    def apply(self, prediction, gold):
        raise NotImplementedError()


class LinearLoss(Loss):
    def apply(self, losses, weights):
        return losses * weights


class Accuracy(Loss):
    def __init__(self, top_k, **kw):
        super(Accuracy, self).__init__(**kw)
        self.top_k = top_k

    def apply(self, prediction, gold):
        if gold.ndim == prediction.ndim:
            gold = T.argmax(gold, axis=-1)
        elif gold.ndim != prediction.ndim - 1:
            raise TypeError('rank mismatch between targets and predictions')

        if self.top_k == 1:
            # standard categorical accuracy
            top = T.argmax(prediction, axis=-1)
            return T.eq(top, gold)
        else:
            # top-k accuracy
            top = T.argsort(prediction, axis=-1)
            # (Theano cannot index with [..., -top_k:], we need to simulate that)
            top = top[[slice(None) for _ in range(top.ndim - 1)] +
                      [slice(-self.top_k, None)]]
            targets = T.shape_padaxis(gold, axis=-1)
            return T.any(T.eq(top, targets), axis=-1)


class SeqAccuracy(Loss):
    def apply(self, probs, gold, mask=None):
        if gold.ndim == probs.ndim:
            gold = T.argmax(gold, axis=-1)
        elif gold.ndim != probs.ndim - 1:
            raise TypeError('rank mismatch between targets and predictions')
        top = T.argmax(probs, axis=-1)
        assert (gold.ndim == 2 and top.ndim == 2)
        assert (mask is None or mask.ndim == 2)
        if mask is not None:
            gold *= mask
            top *= mask
        diff = T.sum(abs(top - gold), axis=1)
        return T.eq(diff, T.zeros_like(diff))


class HingeLoss(Loss):
    def __init__(self, margin=1.0, labelbin=True, **kw):
        super(HingeLoss, self).__init__(**kw)
        self.margin = margin
        self.labelbin = labelbin

    def apply(self, preds, gold):   # preds: (batsize,), gold: (batsize,)
        if self.labelbin is True:
            gold = 2 * gold - 1
        return T.nnet.relu(self.margin - gold * preds)


class LogLoss(Loss):
    """ NOT cross-entropy, BUT log(1+e^(-t*y))"""
    def apply(self, preds, gold):
        return T.nnet.softplus(-gold * preds)


class SquaredLoss(Loss):
    def apply(self, preds, gold):
        return (1 - preds * gold) ** 2


class SquaredError(Loss):
    def apply(self, preds, gold):
        return (preds - gold)**2


class CrossEntropy(Loss):
    def __init__(self, mode="sum", **kw):
        super(CrossEntropy, self).__init__(**kw)
        self.mode = mode

    def apply(self, probs, gold, mask=None):
        mask = probs.mask if mask is None else mask
        if gold.ndim == 1:
            assert(mask is None)
            return T.nnet.categorical_crossentropy(probs, gold) #-tensor.log(probs[tensor.arange(gold.shape[0]), gold])
        elif gold.ndim == 2:    # sequences
            origprobshape = probs.shape
            origprobndim = probs.ndim
            probs = probs.reshape((-1, probs.shape[-1]))
            gold = gold.reshape((-1,))
            seq_ces = T.nnet.categorical_crossentropy(probs, gold)
            o = seq_ces.reshape(origprobshape[:-1], ndim=origprobndim-1)
            o = o * mask if mask is not None else o  # (batsize, seqlen)
            if self.mode == "sum":
                o = o.sum(axis=1)
            elif self.mode == "allmean":
                #print "using allmean"
                rep = o.shape[0]
                div = mask.sum() if mask is not None else o.shape[0] * o.shape[1]
                o = o.sum() / div
                o = T.repeat(o.dimadd(1), rep, axis=0)
            elif self.mode == "rowmean":
                div = mask.sum(axis=1) if mask is not None else o.shape[1]
                o = o.sum(axis=1) / div
            return o  # (batsize,)


class Perplexity(Loss):
    def __init__(self, **kw):
        super(Perplexity, self).__init__(**kw)
        self.ce = CrossEntropy(mode="allmean")

    def apply(self, probs, gold, mask=None):
        mask = probs.mask if mask is None else mask
        perwordce = self.ce(probs, gold, mask=mask)
        ret = 2 ** perwordce
        return ret


class BinaryCrossEntropy(Loss):
    def apply(self, preds, gold):
        return T.nnet.binary_crossentropy(preds, gold)


class BinaryAccuracy(Loss):
    def __init__(self, sep=0, **kw):
        super(BinaryAccuracy, self).__init__(**kw)
        self.sep = sep

    def apply(self, preds, gold):
        return T.eq(preds > self.sep, gold > self.sep)

