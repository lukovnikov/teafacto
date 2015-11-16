__author__ = 'denis'
from teafacto.utils import issequence

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
    def from_ssd(self, ssdfilepath, fill=0.0):
        self = SparseTensor()
        self._fill = fill
        with open(ssdfilepath) as f:
            for line in f:
                ns = line.split(" ")
                idxs = map(lambda x: int(x), ns[:-1])
                val = float(ns[-1])
                self[tuple(idxs)] = val
        return self

    def __init__(self):
        self._dok = {}
        self._maxes = []
        self._fill = 0.0

    @property
    def shape(self):
        return tuple(self._maxes)

    def nonzeros(self, tuples=False):
        if tuples is True:
            return self._dok.keys()
        else:
            return map(list, zip(*self._dok))

    def __getitem__(self, item):
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
        else:
            raise NotImplementedError
        if len(out) == 1:
            out = out[0]
        return out

    def __setitem__(self, idxs, val):
        if len(self._maxes) == 0:
            self._maxes = list(idxs)
        if len(idxs) != len(self._maxes):
            raise Exception("wrong shape of entry")
        self._dok[tuple(idxs)] = val
        for i in range(len(idxs)):
            self._maxes[i] = max(self._maxes[i], idxs[i])


if __name__ == "__main__":
    t = SparseTensor.from_list([
        (0, 0, 0, 1.0),
        (0, 0, 5, 2.0)
    ])
    print t[0, 0, 5]
    print t[[0, 0], [0, 0], [0, 5]]
    print t.nonzeros()