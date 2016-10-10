import teafacto as F
from teafacto.core.base import tensorops as T


class BinaryCrossEntropy(F.Block):
    def apply(self, probs, gold):
        return T.nnet.binary_crossentropy(probs, gold)


class CategoricalCrossEntropy(F.Block):
    def apply(self, probs, gold):
        return T.nnet.categorical_crossentropy(probs, gold)


class Linear(F.Block):
    def apply(self, x, y):
        return x * y


class SquaredError(F.Block):
    def apply(self, x, y):
        return (x - y) ** 2


class SquaredLoss(F.Block):
    def apply(self, x, y):
        return (1 - x * y) ** 2


class HingeLoss(F.Block):
    def __init__(self, margin=1., labelbin=True):
        self.margin = margin
        self.labelbin = labelbin

    def apply(self, preds, gold):
        if self.labelbin is True:
            gold = 2 * gold - 1
        return T.nnet.relu(self.margin - gold * preds)


class LogLoss(F.Block):
    def apply(self, preds, gold):
        return T.nnet.softplus(- gold * preds)


class BinaryAccuracy(F.Block):
    def __init__(self, sep=0, **kw):
        super(BinaryAccuracy, self).__init__(**kw)
        self.sep = sep

    def apply(self, x, y):
        return T.eq(x > self.sep, y > self.sep)


class CategoricalAccuracy(F.Block):
    def __init__(self, top_k=1, **kw):
        super(CategoricalAccuracy, self).__init__(**kw)
        self.top_k = top_k

    def apply(self, predictions, targets):
        if targets.ndim == predictions.ndim:
            targets = T.argmax(targets, axis=-1)
        elif targets.ndim != predictions.ndim - 1:
            raise TypeError("rank mismatch between tagets and prediction")

        if top_k == 1:
            top = T.argmax(predictions, axis=-1)
            return T.eq(top, targets)
        else:
            top = T.argsort(predictions, axis=-1)
            # (Theano cannot index with [..., -top_k:], we need to simulate that)
            top = top[[slice(None) for _ in range(top.ndim - 1)] +
                      [slice(-top_k, None)]]
            targets = T.shape_padaxis(targets, axis=-1)
            return T.any(T.eq(top, targets), axis=-1)