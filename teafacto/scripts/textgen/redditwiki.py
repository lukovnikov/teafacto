import h5py as h5, json, numpy as np
from teafacto.util import argprun, ticktock
from IPython import embed

from teafacto.blocks.seq.encdec import MultiEncDec
from teafacto.blocks.seq.rnn import SeqEncoder
from teafacto.blocks.basic import SMO, VectorEmbed
from teafacto.blocks.seq.attention import Attention
from teafacto.core.base import asblock
from teafacto.blocks.cnn import CNNSeqEncoder


def loaddata(p="../../../data/textgen/redditwiki", maxlen=np.infty, flattenwiki=False):
    tt = ticktock("loader")
    tt.tick("loading")
    redditfile = h5.File("{}/reddit.h5".format(p))
    reddittrain = redditfile["train"][:]
    redditvalid = redditfile["validate"][:]
    reddittest = redditfile["test"][:]
    wikifile = h5.File("{}/wikipedia.h5".format(p))
    wikitrain = wikifile["train"][:]
    wikivalid = wikifile["validate"][:]
    wikitest = wikifile["test"][:]
    wikitrain = wikitrain.reshape((reddittrain.shape[0], 20, wikitrain.shape[1]))
    wikivalid = wikivalid.reshape((redditvalid.shape[0], 20, wikivalid.shape[1]))
    wikitest = wikitest.reshape((reddittest.shape[0], 20, wikitest.shape[1]))

    worddic = json.load(open("{}/dictionary.json".format(p)))
    wd = worddic["word2id"]
    rwd = worddic["id2word"]
    padid = wd["<PAD>"]
    maskid = 0
    wikitrain[wikitrain == padid] = maskid
    wikivalid[wikivalid == padid] = maskid
    wikitest[wikitest == padid] = maskid
    reddittrain[reddittrain == padid] = maskid
    redditvalid[redditvalid == padid] = maskid
    reddittest[reddittest == padid] = maskid
    wd["<MSK>"] = 0
    wd["<UNK>"] = 1
    del wd["NaN"]
    rwd = {v: k for k, v in wd.items()}
    tt.tock("loaded")

    def ps(xl):
        return " ".join([rwd[x] if x in rwd else "<UNK>" for x in list(xl) if x != wd["<MSK>"]])

    tt.tick("transforming")
    traindata, traingold = split_train_gold(reddittrain, startid=wd["<start>"])
    validdata, validgold = split_train_gold(redditvalid, startid=wd["<start>"])
    testdata, testgold = split_train_gold(reddittest, startid=wd["<start>"])
    tt.tock("transformed")

    if flattenwiki:
        tt.tick("flattening wiki")
        wikitrain = flatten_wiki(wikitrain, startid=wd["<start>"], stopid=wd["<end>"])
        wikivalid = flatten_wiki(wikivalid, startid=wd["<start>"], stopid=wd["<end>"])
        wikitest = flatten_wiki(wikitest, startid=wd["<start>"], stopid=wd["<end>"])
        tt.tock("flattened wiki")

    tt.tick("filtering by length")
    traindata, traingold, wikitrain = filter_data(traindata, traingold, wikitrain, maxdatalen=maxlen,
                                                  maxgoldlen=maxlen, maxwikilen=maxlen)
    validdata, validgold, wikivalid = filter_data(validdata, validgold, wikivalid, maxdatalen=maxlen, maxgoldlen=maxlen,
                                                  maxwikilen=maxlen)
    testdata, testgold, wikitest = filter_data(testdata, testgold, wikitest, maxdatalen=maxlen, maxgoldlen=maxlen,
                                                  maxwikilen=maxlen)
    tt.tock("filtering by length")

    traindata, traingold, wikitrain, validdata, validgold, wikivalid, testdata, testgold, wikitest = \
        [x.astype("int32") for x in
     [traindata, traingold, wikitrain, validdata, validgold, wikivalid, testdata, testgold, wikitest]]

    return (traindata, traingold, wikitrain), (validdata, validgold, wikivalid),\
           (testdata, testgold, wikitest), wd, ps


def split_train_gold(seqmat, startid=None):   # extract last comment
    data = np.zeros_like(seqmat)
    gold = np.zeros_like(seqmat)
    datamaxlen = 0
    goldmaxlen = 0
    for i in range(seqmat.shape[0]):
        seqmatrow = seqmat[i]
        j = len(seqmatrow) - 1
        while j > 0:
            if seqmatrow[j] == startid:
                datarow = seqmatrow[:j]
                goldrow = seqmatrow[j:]
                data[i, :len(datarow)] = datarow
                gold[i, :len(goldrow)] = goldrow
                datamaxlen = max(datamaxlen, len(datarow))
                goldmaxlen = max(goldmaxlen, len(goldrow))
                break
            j -= 1
    data = data[:, :datamaxlen]
    gold = gold[:, :goldmaxlen]
    return data, gold


def filter_data(data, gold, wiki, maxdatalen=np.infty, maxgoldlen=np.infty,
                maxwikilen=np.infty, maskid=0):
    if maxdatalen == np.infty and maxgoldlen == np.infty and maxwikilen == np.infty:
        return data, gold, wiki
    datasizes = np.sum(data != maskid, axis=1)
    goldsizes = np.sum(gold != maskid, axis=1)
    wikisizes = np.sum(wiki != maskid, axis=1)
    include = (datasizes <= maxdatalen) * (goldsizes <= maxgoldlen)\
              * (wikisizes <= maxwikilen)

    data = data[include, :maxdatalen]
    gold = gold[include, :maxgoldlen]
    wiki = wiki[include, :maxwikilen]
    return data, gold, wiki


def flatten_wiki(wikitensor, maskid=0, startid=1, stopid=2):   # (numsam, numsent, sentlen) -> (numsam, numsent * sentlen)
    out = np.zeros((wikitensor.shape[0], wikitensor.shape[1] * wikitensor.shape[2]), dtype="int32")
    maxoutlen = 0
    for i in range(wikitensor.shape[0]):
        outrow = []
        for j in range(wikitensor.shape[1]):
            k = wikitensor.shape[2] - 1
            while k > 0:
                if wikitensor[i, j, k] != maskid:
                    outrow += [startid] + list(wikitensor[i, j, :k+1]) + [stopid]
                    break
                k -= 1
        out[i, :len(outrow)] = outrow
        maxoutlen = max(maxoutlen, len(outrow))
    return out


def run(p="../../../data/textgen/redditwiki",       # path used by loaddata
        embdim=100,         # word embeddings dimensions of all inputs
        encdim=300,         # internal dimensions of all encoders
        smodim=200,         # softmax output layer input dimension
        decdim=500,         # internal dimension of decoder
        lr=0.5,             # learning rate
        gradnorm=5.,        # gradient normalization
        numbats=200,        # number of batches
        epochs=100,         # number of epochs
        inspectdata=False,  # turn switch to get interactive shell for inspecting loaded data
        srcenc="cnn",       # or "rnn" - what kind of encoding network to use
        usestrides=False,   # whether to use striding in CNN encoder <- reduces number of attentions (makes it lighter, not sure about performance yet)
        maxlen=600,         # maximum length of input
        dropout=0.2,        # dropout level
        posembdim=50,       # position embedding dimensions, only with CNN encoder
        sameencoder=False,  # use the same encoding network for all inputs
        debugvalid=True,
        ):
    (traindata, traingold, trainwiki), (validdata, validgold, validwiki), \
    (testdata, testgold, testwiki), wd, ps \
        = loaddata(p, maxlen=maxlen, flattenwiki=True)
    print "retained {} examples".format(len(traindata))
    print ps(trainwiki[1])
    print ps(traindata[1])
    print ps(traingold[1])

    if inspectdata:
        embed()

    numwords = max(wd.values()) + 1
    maskid = 0
    splitdim = int(round(encdim / 2.))

    emb = VectorEmbed(numwords, embdim, maskid=maskid)

    if srcenc == "rnn":
        encoder_one = SeqEncoder.fluent() \
            .setembedder(emb) \
            .addlayers([splitdim], bidir=True).addlayers([encdim]) \
            .make().all_outputs()
        if not sameencoder:
            encoder_two = SeqEncoder.fluent() \
                .setembedder(emb) \
                .addlayers([splitdim], bidir=True).addlayers([encdim]) \
                .make().all_outputs()
        else:
            encoder_two = encoder_one

    elif srcenc == "cnn":
        windows = [3, 4, 5, 6]
        if usestrides:
            strides = [1, 1, 2, 3]
        else:
            strides = 1
        encoder_one = CNNSeqEncoder(inpemb=emb, windows=windows, stride=strides,
                                    innerdim=[encdim, encdim, encdim, encdim],
                                    dropout=dropout,
                                    posembdim=posembdim, numpos=maxlen).all_outputs()
        if not sameencoder:
            encoder_two = CNNSeqEncoder(inpemb=emb, windows=windows, stride=strides,
                                        innerdim=[encdim, encdim, encdim, encdim],
                                        dropout=dropout,
                                        posembdim=posembdim, numpos=maxlen).all_outputs()
        else:
            encoder_two = encoder_one

    else:
        raise Exception("unknown srcenc option")

    splitters = (asblock(lambda x: x[:, :, :splitdim]), asblock(lambda x: x[:, :, encdim-splitdim:]))
    attention_one = Attention(splitters=splitters)
    splitters = (asblock(lambda x: x[:, :, :splitdim]), asblock(lambda x: x[:, :, encdim-splitdim:]))
    attention_two = Attention(splitters=splitters)

    smo = SMO(smodim, numwords)

    slices = (splitdim, splitdim)

    m = MultiEncDec(encoders=[encoder_one, encoder_two],
                    indim=(encdim-splitdim)*2 + embdim,
                    slices=slices,
                    attentions=[attention_one, attention_two],
                    inpemb=emb,
                    smo=smo,
                    innerdim=[decdim, decdim],
                    dropout_in=dropout)


    def get_perplexity():   # TODO
        runningperplexitysum = [0]
        numperplexitywords = [0]

        def perplexity(ingold, data, wiki, outgold):
            predprobs = m.predict(ingold, data, wiki)
            mask = outgold != maskid
            rightprobs = predprobs[outgold]
            ces = -np.log(rightprobs)
            ces = ces * mask
            runningperplexitysum[0] += np.sum(ces)
            numperplexitywords[0] += np.sum(mask)
            ret = runningperplexitysum[0] / numperplexitywords[0]
            embed()
            return ret
        return perplexity

    m.train([traingold[:, :-1], traindata, trainwiki], traingold[:, 1:])\
        .adadelta(lr=lr).cross_entropy().grad_total_norm(gradnorm)\
        .validate_on([validgold[:, :-1], validdata, validwiki], validgold[:, 1:])\
            .cross_entropy()\
        .autosaveit().takebest(save=True)\
        .train(numbats=numbats, epochs=epochs, _skiptrain=debugvalid)


if __name__ == "__main__":
    argprun(run)