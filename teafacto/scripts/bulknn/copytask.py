from teafacto.blocks.seq.memnn import SimpleBulkNN
from teafacto.util import argprun
import numpy as np


def run(epochs=100, lr=1.):
    inpvocsize = 100
    outvocsize = 100
    inpembdim = 20
    memembdim = 22
    inpdim = [30]
    memdim = [30]
    memlen = 10
    writedim = 19
    posvecdim = 10
    lastcoredim = inpdim[-1] + memdim[-1] + memdim[-1] \
                  + writedim + 1 + 1 + posvecdim * 3
    coredims = [100, lastcoredim]     # last dim must match interface when explicit interface

    maskid = -1

    m = SimpleBulkNN(inpvocsize=inpvocsize,
                     inpembdim=inpembdim,
                     inpencinnerdim=inpdim,
                     memvocsize=outvocsize,
                     memembdim=memembdim,
                     memencinnerdim=memdim,
                     memlen=memlen,
                     coredims=coredims,
                     explicit_interface=True,
                     write_value_dim=writedim,
                     posvecdim=posvecdim,
                     nsteps=20,
                     maskid=maskid,
                     memsamplemethod="gumbel",      # or "gumbel"
                     dropout=0.3,
                     memsampletemp=0.3)

    data = np.random.randint(1, 100, (1000, 10))
    m.train([data], data).cross_entropy().adadelta(lr=lr)\
        .train(epochs=epochs, numbats=10)


if __name__ == "__main__":
    argprun(run)