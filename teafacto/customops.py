from theano.gof import Op
import theano
from theano import tensor as T
import numpy as np
from IPython import embed

from teafacto.core.base import Block


class STE_Threshold(Block):
    def __init__(self, threshold=0.0, **kw):
        self._inner = _STE_Threshold(threshold=threshold)
        super(STE_Threshold, self).__init__(**kw)

    def _apply(self, x):
        return self._inner(x)


class _STE_Threshold(Op):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        super(_STE_Threshold, self).__init__()

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage, params=None):
        x = inputs[0]
        z = output_storage[0]
        ret = (x > self.threshold).astype(x.dtype)
        z[0] = ret

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        return output_grads


class STE_MaxHot(Block):
    def __init__(self, axes=-1, **kw):
        self._inner = _STE_MaxHot(axes=axes)
        super(STE_MaxHot, self).__init__(**kw)

    def _apply(self, x):
        return self._inner(x)


class _STE_MaxHot(Op):
    def __init__(self, axes=(-1,)):
        if isinstance(axes, int):
            axes = (axes,)
        assert(isinstance(axes, tuple))
        self.axes = axes
        super(_STE_MaxHot, self).__init__()

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage, params=None):
        x = inputs[0]
        z = output_storage[0]
        maxes = np.max(x, axis=self.axes, keepdims=True)
        ret = (x == maxes).astype(x.dtype)
        z[0] = ret

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        return output_grads


if __name__ == "__main__":
    x = theano.shared(np.random.random((10, 5)))
    x = x ** 2
    y = STE_Threshold(0.3)(x)
    c = T.sum(y)
    res = y.eval()
    #print res
    ste_grads = T.grad(c, [x])
    y_d = (x > 0.3) * 1.
    c_d = T.sum(y_d)
    noste_grads = T.grad(c_d, [x])
    #embed()
    y = STE_MaxHot()(x ** 2)
    c = T.sum(y)
    res = y.eval()
    ste_grads = T.grad(c, [x])
    embed()

