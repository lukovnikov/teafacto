from unittest import TestCase
from teafacto.blocks.stack import *
from teafacto.blocks.rnu import *


class TestStack(TestCase):
    def setUp(self):
        self.dims = [50, 20, 30, 40]
        grus = [GRU(dim=self.dims[i], innerdim=self.dims[i+1]) for i in range(len(self.dims)-1)]
        self.s = stack(*grus)
        self.paramnames = ["um", "wm", "uhf", "whf", "u", "w", "bm", "bhf", "b"]

    def test_rnu_stack_parameter_propagation(self):
        o = self.s(Input(ndim=3, dtype="float32", name="stack_input"))
        allps = [x.name for x in o.allparams]
        for paramname in self.paramnames:
            self.assertEqual(allps.count(paramname), len(self.dims)-1)

