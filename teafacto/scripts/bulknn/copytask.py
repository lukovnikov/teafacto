from teafacto.blocks.seq.memnn import SimpleBulkNN, SimpleMemNN
from teafacto.core.base import asblock
from teafacto.util import argprun
import numpy as np
from IPython import embed


def runmem(epochs=100, lr=1.):
    inpvocsize = 50
    inpembdim = 10
    maskid=-1
    posvecdim = 10
    memdim = 12
    memlen = 17
    outdim = 10
    outvocsize = 50

    lastcoredim = outdim + memdim * 3 + posvecdim * 2 + 1 + 1
    coredims = [40, lastcoredim]

    m = SimpleMemNN(inpvocsize=inpvocsize, inpembdim=inpembdim,
                    maskid=maskid, posvecdim=posvecdim,
                    coredims=coredims, memdim=memdim, memlen=memlen,
                    outdim=outdim, outvocsize=outvocsize)


    seqlen = 10
    b = asblock(lambda x: m(x)[:, seqlen+1:])

    origdata = np.random.randint(1, inpvocsize, (1000, seqlen))
    data = origdata
    data = np.concatenate([data, np.zeros((1000, 1)).astype("int32"), data], axis=1)

    b.train([data], origdata).cross_entropy().adadelta(lr=lr) \
        .train(epochs=epochs, numbats=10)

    origpreddata = np.random.randint(1, inpvocsize, (50, seqlen))
    preddata = origpreddata
    preddata = np.concatenate([preddata, np.zeros((50, 1)).astype("int32"), preddata], axis=1)

    pred = b.predict(preddata)
    print np.argmax(pred, axis=2)
    embed()



def runbulk(epochs=100, lr=1.):
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
                     memsampletemp=0.3)

    data = np.random.randint(1, 100, (1000, 10))
    m._return_all_mems = True
    pred, all = m.predict(data)
    for i in range(all.shape[0]):
        print np.argmax(all[i], axis=2)

    m._return_all_mems = False
    m.train([data], data).cross_entropy().adadelta(lr=lr)\
        .train(epochs=epochs, numbats=10)


if __name__ == "__main__":
    argprun(runmem)