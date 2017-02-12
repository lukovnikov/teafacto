from teafacto.blocks.seq.memnn import SimpleBulkNN, SimpleMemNN, SimpleTransMemNN
from teafacto.core.base import asblock
from teafacto.util import argprun
import numpy as np
from IPython import embed


def runmem(epochs=100, lr=1.,
           mode="copy",     # or "sort" or "reverse"
           seqlen=10,
           numsam=5000,
           vocsize=20,
           embdim=10,
           memdim=20,
           dropout=0.0,
           transmode=False,
           rnnposgen=False,         # this works, but is slower
           addrsample=False,        # this is shit, validation loss stays very high
                                    # maybe because during prediction, is replaced
                                    # by normal unsharpened softmax
                                    # TODO: need to make sure that GumbelSoftmax
                                    # during prediction is as sharp as the
                                    # sampled Gumbel (or just use sampled during pred)
                                    # QUESTION: does training w. gumbel force
                                    # the network to produce sharper softmaxes?
           ):
    inpvocsize = vocsize
    inpembdim = embdim
    maskid = -1
    posvecdim = memdim
    memlen = seqlen*2
    outdim = embdim
    outvocsize = vocsize

    lastcoredim = outdim + memdim * 3 + posvecdim * 2 + 1 + 1
    coredims = [lastcoredim, lastcoredim]

    addrsample = "gumbel" if addrsample else None

    if not transmode:
        m = SimpleMemNN(inpvocsize=inpvocsize, inpembdim=inpembdim,
                        maskid=maskid, posvecdim=posvecdim,
                        coredims=coredims, memdim=memdim, memlen=memlen,
                        outdim=outdim, outvocsize=outvocsize,
                        dropout_h=dropout,
                        rnn_pos_gen=rnnposgen, addr_sampler=addrsample,
                        addr_sample_temperature=0.2)
        b = asblock(lambda x: m(x)[:, seqlen:-1])
    else:
        b = SimpleTransMemNN(inpvocsize=inpvocsize, inpembdim=inpembdim,
                        maskid=maskid, posvecdim=posvecdim,
                        coredims=coredims, memdim=memdim, memlen=memlen,
                        outdim=outdim, outvocsize=outvocsize, outembdim=inpembdim,
                        dropout_h=dropout, rnn_pos_gen=rnnposgen,
                        addr_sampler=addrsample, addr_sample_temperature=0.2)

    origdata = np.random.randint(1, inpvocsize, (numsam, seqlen))
    data = origdata

    def gettargetdata(inpd):
        if mode == "copy":
            return inpd
        elif mode == "sort":
            return np.sort(inpd, axis=1)
        elif mode == "revert":
            raise NotImplementedError("not implemented yet")
        else:
            raise Exception("unrecognized mode")

    if transmode:
        inpdata = data
        outdata = gettargetdata(data)
        ioutdata = np.concatenate([
            np.zeros((outdata.shape[0],))[:, np.newaxis].astype("int32"),
            inpdata[:, :-1]], axis=1)
        b.train([data, ioutdata], outdata).cross_entropy().adadelta(lr=lr) \
            .split_validate(5).cross_entropy().seq_accuracy() \
            .train(epochs=epochs, numbats=10)
    else:
        data = np.concatenate([data, np.zeros((numsam, 1)).astype("int32"), gettargetdata(data)], axis=1)
        b.train([data], origdata).cross_entropy().adadelta(lr=lr)\
            .split_validate(5).cross_entropy().seq_accuracy() \
            .train(epochs=epochs, numbats=10)

        origpreddata = np.random.randint(1, inpvocsize, (25, seqlen))
        preddata = origpreddata
        preddata = np.concatenate([preddata, np.zeros((25, 1)).astype("int32"), gettargetdata(preddata)], axis=1)

        pred = b.predict(preddata)
        print np.argmax(pred, axis=2)
    #embed()



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