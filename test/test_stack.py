from unittest import TestCase

import numpy as np

from teafacto.blocks.rnu import GRU
from teafacto.blocks.rnn import RecurrentStack, ReccableStack, SimpleSeqTransDec
from teafacto.core.stack import stack
from teafacto.core.base import Input, param, asblock, tensorops as T
from teafacto.blocks.basic import Softmax, VectorEmbed
import theano


class TestRecurrentStack(TestCase):
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


class TestRecurrentStackRecappl(TestCase):
    def test_recappl(self):
        batsize = 100
        self.dims = [50, 20, 30, 40]
        recstack = RecurrentStack(*[GRU(dim=self.dims[i], innerdim=self.dims[i+1]) for i in range(len(self.dims)-1)])
        recapplestates, _ = recstack.get_init_info(batsize)
        for recapplestate in recapplestates:
            print recapplestate.d.eval().shape
        inp = Input(ndim=2, dtype="float32", name="stack_input")
        stateinps = [Input(ndim=x.d.ndim, dtype="float32") for x in recapplestates]
        allinps = [inp] + stateinps
        out, states, tail = recstack.recappl(inp, stateinps)
        allouts = out + states
        assert(len(tail) == 0)
        f = theano.function(inputs=[x.d for x in allinps], outputs=[x.d for x in allouts])

        statevals = [recapplestate.d.eval() for recapplestate in recapplestates]
        for i in range(3):
            inpval = np.random.random((batsize, 50)).astype("float32")
            inpvals = [inpval] + statevals
            outpvals = f(*inpvals)
            print [x.shape for x in outpvals]
            statevals = outpvals[1:]


class TestSeqTransDecRecappl(TestCase):     # TODO: move this test
    def test_recappl_shapes(self):
        batsize = 100
        recstack = SimpleSeqTransDec(indim=200, outdim=50, inpembdim=20, outembdim=20, innerdim=[40, 30])
        recapplestates, _ = recstack.get_init_info(batsize)
        inpinp = Input(ndim=1, dtype="int32", name="stack_input")
        outinp = Input(ndim=1, dtype="int32", name="stack_input_2")
        stateinps = [Input(ndim=x.d.ndim, dtype="float32") for x in recapplestates]
        allinps = [inpinp, outinp] + stateinps
        out, states, tail = recstack.recappl([inpinp, outinp], stateinps)
        allouts = out + states
        assert(len(tail) == 0)
        f = theano.function(inputs=[x.d for x in allinps], outputs=[x.d for x in allouts])

        statevals = [recapplestate.d.eval() for recapplestate in recapplestates]
        for i in range(5):
            inpval = np.random.randint(0, 200, (batsize,)).astype("int32")
            inpval2 = np.random.randint(0, 50, (batsize,)).astype("int32")
            inpvals = [inpval, inpval2] + statevals
            outpvals = f(*inpvals)
            self.assertEqual(outpvals[0].shape, (batsize, 50))
            for x, y in zip([40, 30], outpvals[1:]):
                self.assertEqual(y.shape, (batsize, x))
            print [x.shape for x in outpvals]
            statevals = outpvals[1:]



class TestBlockStack(TestCase):
    def setUp(self):
        dim=50
        self.vocabsize=2000
        data = np.arange(0, self.vocabsize).astype("int32")
        self.O = param((dim, self.vocabsize)).uniform()
        self.W = VectorEmbed(indim=self.vocabsize, dim=50)
        self.out = stack(self.W,
              asblock(lambda x: T.dot(self.O, x)),
              Softmax())(Input(ndim=1, dtype="int32"))

    def test_param_propagation(self):
        self.assertSetEqual(set(self.out.allparams), {self.O, self.W.W})

