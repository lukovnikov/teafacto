from teafacto.blocks.core import *
from teafacto.blocks.datafeed import *
from teafacto.blocks.model import *
from teafacto.blocks.trainer import *
from teafacto.core.utils import argparsify


class AutoEncoderModel(Model):
    def __init__(self, dim=50, **kw):
        super(AutoEncoderModel, self).__init__(**kw)
        self.dim = dim
        self.W = self.add_param(param((10, 10)).uniform()).d

    def initinputs(self):
        return [input(ndim=1, dtype="int32")]

    def apply(self, inp):
        pass


def run(
        epochs=10,
    ):
    return epochs


if __name__ == "__main__":
    run(**argparsify(run))
