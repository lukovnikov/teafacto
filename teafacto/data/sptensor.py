__author__ = 'denis'
from teafacto.utils import issequence
import numpy as np
from collections import OrderedDict

class SparseTensor(object):

    @staticmethod
    def from_list(list, fill=0.0):
        self = SparseTensor()
        self._fill = fill
        for ns in list:
            idxs = map(lambda x: int(x), ns[:-1])
            val = float(ns[-1])
            self[tuple(idxs)] = val
        return self

    @staticmethod
    def from_ssd(ssdfilepath, fill=0.0):
        self = SparseTensor()
        self._fill = fill
        with open(ssdfilepath) as f:
            for lc, line in enumerate(f, start=1):
                ns = line.split(" ")
                idxs = map(lambda x: int(x), ns[:-1])
                val = float(ns[-1])
                self[tuple(idxs)] = val
        return self

    def __init__(self):
        self._dok = OrderedDict()
        self._maxes = []
        self._fill = 0.0

    @property
    def shape(self):
        return tuple(map(lambda x: x+1, self._maxes))

    def __len__(self):
        return len(self._dok)

    @property
    def numdims(self):
        ret = 0
        for x in self._dok:
            ret = len(x)
            break
        return ret

    def count_nonzeros(self):
        return len(self._dok)

    def nonzeros(self, tuples=False, withvals=False):
        if tuples is True:
            if withvals is True:
                return self._dok
            else:
                return self._dok.keys()
        else:
            if withvals is True:
                return np.asarray(map(list, zip(*map(lambda (x, y): (x+(y,)), self._dok.items()))))
            else:
                return np.asarray(map(list, zip(*self._dok.keys())))

    def __getitem__(self, item):
        print "item " + str(item)
        out = []
        if isinstance(item, tuple): # tuple of lists #TODO support slices
            def mapper(s):
                if not issequence(s):
                    if not isinstance(s, slice):
                        return [s]
                    else:
                        raise NotImplementedError
                else:
                    return s
            item = map(mapper, item)
            idxs = zip(*item)
            for idx in idxs:
                if idx in self._dok:
                    out.append(self._dok[idx])
                else:
                    out.append(self._fill)
        elif isinstance(item, list):
            out = self._dok.values()[list]
        if len(out) == 1:
            out = out[0]
        return out

    @property
    def keys(self):
        return SparseTensorKeys(self._dok.keys())

    def __setitem__(self, idxs, val):
        if len(self._maxes) == 0:
            self._maxes = list(idxs)
        if len(idxs) != len(self._maxes):
            return
            raise Exception("wrong shape of entry")
        self._dok[tuple(idxs)] = val
        for i in range(len(idxs)):
            self._maxes[i] = max(self._maxes[i], idxs[i])

    def threshold(self, th):
        for k, v in self._dok.items():
            if v < th:
                del self._dok[k]


class SparseTensorKeys(object):
    def __init__(self, lok):
        self.lok = np.asarray(lok)

    def __getitem__(self, item):
        return self.lok[item].T

if __name__ == "__main__":
    t = SparseTensor.from_list([
        (0, 0, 0, 1.0),
        (0, 0, 5, 2.0),
        (4, 3, 6, 3.0)
    ])

    print "lok: \n" + str(t.keys[0, 1][0])