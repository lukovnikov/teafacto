from teafacto.util import argprun, ticktock
from teafacto.blocks.seqproc import SimpleSeq2Vec, SeqEncDecAtt, SimpleSeqEncDecAtt, SeqUnroll
from teafacto.blocks.match import SeqMatchScore, GenDotDistance
from teafacto.blocks.basic import VectorEmbed
from teafacto.core.base import Val, tensorops as T, Block
import pickle, numpy as np, sys, os
from IPython import embed
from teafacto.search import SeqEncDecSearch
from teafacto.eval.metrics import ClassAccuracy
from teafacto.modelusers import RecPredictor


def readdata(mode, testcans=None, debug=False, specids=False, usetypes=False):  # if none, included in file

    if debug:
        testcans = None
    if mode == "char":
        if debug:
            p = "../../../data/simplequestions/datamat.char.mini.pkl"
        else:
            p = "../../../data/simplequestions/datamat.char.mem.fb2m.pkl"
    elif mode == "word":
        p = "../../../data/simplequestions/datamat.word.mem.fb2m.pkl"
    elif mode == "charword":
        p = "../../../data/simplequestions/datamat.charword.mem.fb2m.pkl"
    else:
        raise Exception("unknown mode")
    if usetypes:
        enttyp = {}
        typdic = {"None": 0}
        with open("../../../data/simplequestions/datamat.fb2m.types.map") as f:
            for line in f:
                e, t = line[:-1].split("\t")
                enttyp[e] = t
    x = pickle.load(open(p))
    worddic = x["worddic"] if mode == "word" else x["chardic"]
    worddic = {k: v+1 for k, v in worddic.items()}
    worddic["|"] = 0
    worddic2 = x["worddic"] if mode == "charword" else None
    entdic = x["entdic"]
    entdic = {k: v+1 for k, v in entdic.items()}
    entdic["|"] = 0
    numents = x["numents"]+1

    def shiftidxs(mat, shift=1, mask=-1):
        shifted = mat + shift
        shifted[shifted == (mask + shift)] = mask
        return shifted

    entmat = shiftidxs(x["entmat"])
    addtoentmat = -np.ones_like(entmat[np.newaxis, 0], dtype="int32")
    addtoentmat[0, 0] = 0
    entmat = np.concatenate([addtoentmat, entmat], axis=0)

    if specids: # prepend special id's (non-unique for entities, unique for predicates, unique for others)
        prependmat = np.zeros((entmat.shape[0], 1), dtype="int32")
        mid = 1
        if usetypes:
            reventdic = {v: k for k, v in entdic.items()}
            for i in range(1, numents):
                t = enttyp[reventdic[i]]
                if t not in typdic:
                    typdic[t] = len(typdic)
                    mid = len(typdic)
                prependmat[i] = typdic[t]
        # same ids for all entitites
        # unique predicate ids
        for i in range(numents, entmat.shape[0]):
            prependmat[i] = mid
            mid += 1
        # unique other ids
        prependmat[0] = mid
        entmat = np.concatenate([prependmat, entmat], axis=1)
        if debug:
            pass
            #embed()

    #embed()

    def shiftidxstup(t, shift=1, mask=-1):
        return tuple([shiftidxs(te, shift=shift, mask=mask) for te in t])

    train = shiftidxstup(x["train"])
    valid = shiftidxstup(x["valid"])
    test  = shiftidxstup(x["test"])

    if testcans is None:
        canids = x["testcans"]
    else:
        canids = pickle.load(open(testcans))
    for i in range(len(canids)):  # offset existing canids
        canids[i] = [canid + 1 for canid in canids[i]]
    for canidl in canids:  # these are already offset
        canidl.extend(range(numents, entmat.shape[0]))  # include all relations

    return train, valid, test, worddic, entdic, entmat, numents, canids


class SeqEncDecRankSearch(SeqEncDecSearch):
    def __init__(self, model, canenc, scorer, agg, beamsize=1, *buildargs, **kw):
        super(SeqEncDecRankSearch, self).__init__(model, beamsize, *buildargs, **kw)
        self.scorer = scorer
        self.canenc = canenc
        self.agg = agg
        self.tt = ticktock("RankSearch")

    def decode(self, inpseq, initsymbolidx, maxlen=100, candata=None,
               canids=None, transform=None, debug=False):
        assert(candata is not None and canids is not None)

        self.mu.setbuildargs(inpseq)
        self.mu.settransform(transform)
        stop = False
        j = 0
        curout = np.repeat([initsymbolidx], inpseq.shape[0]).astype("int32")
        accscores = []
        outs = []
        while not stop:
            curvectors = self.mu.feed(curout)
            curout = np.ones_like(curout, dtype=curout.dtype) * initsymbolidx
            accscoresj = np.zeros((inpseq.shape[0],))
            self.tt.tick()
            for i in range(curvectors.shape[0]):    # for each example, find the highest scoring suited cans and their scores
                #print len(canids[i])
                if len(canids[i]) == 0:
                    curout[i] = -1
                else:
                    canidsi = canids[i]
                    candatai = candata[canidsi]
                    canrepsi = self.canenc.predict(candatai)
                    curvectori = np.repeat(curvectors[np.newaxis, i, ...], canrepsi.shape[0], axis=0)
                    scoresi = self.scorer.predict(curvectori, canrepsi)
                    curout[i] = canidsi[np.argmax(scoresi)]
                    accscoresj[i] += np.max(scoresi)
                    if debug:
                        print i, sorted(zip(canidsi, scoresi), key=lambda (x, y): y, reverse=True)
                        print sorted(filter(lambda (x, y): x < 4711, zip(canidsi, scoresi)), key=lambda (x, y): y, reverse=True)
                        print sorted(filter(lambda (x, y): x >= 4711, zip(canidsi, scoresi)), key=lambda (x, y): y,
                                     reverse=True)
                    #embed()
                self.tt.progress(i, curvectors.shape[0], live=True)
            accscores.append(accscoresj[:, np.newaxis])
            outs.append(curout)
            j += 1
            stop = j == maxlen
            self.tt.tock("done one timestep")
        accscores = np.sum(np.concatenate(accscores, axis=1), axis=1)
        ret = np.stack(outs).T
        assert (ret.shape[0] == inpseq.shape[0] and ret.shape[1] <= maxlen)
        return ret, accscores


class FullRankEval(object):
    def __init__(self):
        self.metrics = {"all": ClassAccuracy(),
                        "subj": ClassAccuracy(),
                        "pred": ClassAccuracy()}

    def eval(self, pred, gold, debug=False):
        for i in range(pred.shape[0]):
            self.metrics["all"].accumulate(gold[i], pred[i])
            if debug == "subj":
                self.metrics["subj"].accumulate(gold[i][0], pred[i][0])
                self.metrics["pred"].accumulate(gold[i][0], gold[i][0])     #dummy
            elif debug == "pred":
                self.metrics["pred"].accumulate(gold[i][0], pred[i][0])
                self.metrics["subj"].accumulate(gold[i][0], gold[i][0])     #dummy
            else:
                self.metrics["subj"].accumulate(gold[i][0], pred[i][0])
                self.metrics["pred"].accumulate(gold[i][1], pred[i][1])
        return self.metrics


class EntEmbEnc(Block):
    def __init__(self, enc, vocsize, embdim, **kw):
        super(EntEmbEnc, self).__init__(**kw)
        self.enc = enc
        self.emb = VectorEmbed(vocsize, embdim)

    def apply(self, x, mask=None):
        enco = self.enc(x[:, 1:], mask=mask)
        embo = self.emb(x[:, 0])
        ret = T.concatenate([enco, embo], axis=1)                   # (?, encdim+embdim)
        return ret

    @property
    def outdim(self):
        return self.enc.outdim + self.emb.outdim


class EntEmbEncRep(EntEmbEnc):
    ''' Replaces encoding with embedding based on repsplit index '''
    def __init__(self, enc, vocsize, repsplit, **kw):
        super(EntEmbEncRep, self).__init__(enc, vocsize, enc.outdim, **kw)
        self.split = repsplit

    def apply(self, x, mask=None):
        enco = self.enc(x[:, 1:], mask=mask)
        embo = self.emb(x[:, 0])
        repl = x[:, 0] >= self.split
        mask = None
        ret = (1 - mask) * enco + mask * embo

def shiftdata(d):  # idx (batsize, seqlen)
    ds = np.zeros_like(d)
    ds[:, 1:] = d[:, :-1]
    return ds


def run(
        epochs=50,
        mode="char",    # "char" or "word" or "charword"
        numbats=100,
        lr=0.1,
        wreg=0.000001,
        bidir=False,
        layers=1,
        innerdim=200,
        embdim=100,
        negrate=1,
        margin=1.,
        hingeloss=False,
        debug=False,
        preeval=False,
        sumhingeloss=False,
        checkdata=False,        # starts interactive shell for data inspection
        printpreds=False,
        subjpred=False,
        predpred=False,
        specemb=-1,
        balancednegidx=False,
        usetypes=False,
    ):
    if debug:       # debug settings
        sumhingeloss = True
        numbats = 10
        lr = 0.02
        epochs = 10
        printpreds = True
        whatpred = "all"
        if whatpred == "pred":
            predpred = True
        elif whatpred == "subj":
            subjpred = True
        preeval = True
        specemb = 100
        margin = 1.
        balancednegidx = True
        #usetypes=True
    # load the right file
    tt = ticktock("script")
    specids = specemb > 0
    tt.tick()
    (traindata, traingold), (validdata, validgold), (testdata, testgold), \
    worddic, entdic, entmat, relstarts, canids\
        = readdata(mode, testcans="testcans.pkl", debug=debug, specids=specids, usetypes=usetypes)
    entmat = entmat.astype("int32")

    #embed()

    if subjpred is True and predpred is False:
        traingold = traingold[:, [0]]
        validgold = validgold[:, [0]]
        testgold = testgold[:, [0]]
    if predpred is True and subjpred is False:
        traingold = traingold[:, [1]]
        validgold = validgold[:, [1]]
        testgold = testgold[:, [1]]


    if checkdata:
        rwd = {v: k for k, v in worddic.items()}
        red = {v: k for k, v in entdic.items()}
        def p(xids):
            return (" " if mode == "word" else "").join([rwd[xid] if xid > -1 else "" for xid in xids])
        embed()

    reventdic = {v: k for k, v in entdic.items()}
    revworddic = {v: k for k, v in worddic.items()}
    print traindata.shape, traingold.shape, testdata.shape, testgold.shape

    tt.tock("data loaded")

    # *data: matrix of word ids (-1 filler), example per row
    # *gold: vector of true entity ids
    # entmat: matrix of word ids (-1 filler), entity label per row, indexes according to *gold
    # *dic: from word/ent-fbid to integer id, as used in data

    numwords = max(worddic.values()) + 1
    numents = max(entdic.values()) + 1
    print "%d words, %d entities" % (numwords, numents)

    if bidir:
        encinnerdim = [innerdim / 2] * layers
    else:
        encinnerdim = [innerdim] * layers

    memembdim = embdim
    memlayers = layers
    membidir = bidir
    if membidir:
        innerdim = [innerdim/2]*memlayers
    else:
        innerdim = [innerdim]*memlayers

    entenc = SimpleSeq2Vec(indim=numwords,
                         inpembdim=memembdim,
                         innerdim=innerdim,
                         maskid=-1,
                         bidir=membidir)

    if specids:     # include vectorembedder
        numentembs = len(np.unique(entmat[:, 0]))
        entenc = EntEmbEnc(entenc, numentembs, specemb)
        # adjust params for enc/dec construction
        #encinnerdim[-1] += specemb
        #innerdim[-1] += specemb

    encdec = SimpleSeqEncDecAtt(inpvocsize=numwords, inpembdim=embdim,
                    encdim=encinnerdim, bidir=bidir, outembdim=entenc,
                    decdim=innerdim, outconcat=True, vecout=True,
                    statetrans=True)

    scorerargs = ([encdec, SeqUnroll(entenc)],
                  {"argproc": lambda x, y, z: ((x, y), (z,)),
                   "scorer": GenDotDistance(encinnerdim[-1]+innerdim[-1], entenc.outdim)})
    if sumhingeloss:
        scorerargs[1]["aggregator"] = lambda x: x  # no aggregation of scores
    scorer = SeqMatchScore(*scorerargs[0], **scorerargs[1])

    #scorer.save("scorer.test.save")

    # TODO: below this line, check and test
    class PreProc(object):
        def __init__(self, entmat):
            self.f = PreProcE(entmat)

        def __call__(self, encdata, decsg, decgold):        # gold: idx^(batsize, seqlen)
            return (encdata, self.f(decsg), self.f(decgold)), {}

    class PreProcE(object):
        def __init__(self, entmat):
            self.em = Val(entmat)

        def __call__(self, x):
            return self.em[x]

    transf = PreProc(entmat)

    class NegIdxGen(object):
        def __init__(self, rng, midsplit=None):
            self.min = 0
            self.max = rng
            self.midsplit = midsplit

        def __call__(self, datas, sgold, gold):    # the whole target sequence is corrupted, corruption targets the whole set of entities and relations together
            if self.midsplit is None or not balancednegidx:
                return datas, sgold, np.random.randint(self.min, self.max, gold.shape).astype("int32")
            else:
                entrand = np.random.randint(self.min, self.midsplit, gold.shape)
                relrand = np.random.randint(self.midsplit, self.max, gold.shape)
                mask = np.random.randint(0, 2, gold.shape)
                ret = entrand * mask + relrand * (1 - mask)
                return datas, sgold, ret.astype("int32")

    # !!! MASKS ON OUTPUT SHOULD BE IMPLEMENTED FOR VARIABLE LENGTH OUTPUT SEQS
    obj = lambda p, n: n - p
    if hingeloss:
        obj = lambda p, n: (n - p + margin).clip(0, np.infty)
    if sumhingeloss:    #
        obj = lambda p, n: T.sum((n - p + margin).clip(0, np.infty), axis=1)

    traingoldshifted = shiftdata(traingold)
    validgoldshifted = shiftdata(validgold)

    #embed()
    # eval
    if preeval:
        tt.tick("pre-evaluating")
        s = SeqEncDecRankSearch(encdec, entenc, scorer.s, scorer.agg)
        eval = FullRankEval()
        pred, scores = s.decode(testdata, 0, testgold.shape[1],
                                candata=entmat, canids=canids,
                                transform=transf.f, debug=printpreds)
        evalres = eval.eval(pred, testgold, debug=debug)
        for k, evalre in evalres.items():
            print("{}:\t{}".format(k, evalre))
        tt.tock("pre-evaluated")

    negidxgenargs = ([numents], {"midsplit": relstarts})
    if debug:
        pass
        #negidxgenargs = ([numents], {})

    tt.tick("training")
    nscorer = scorer.nstrain([traindata, traingoldshifted, traingold]).transform(transf) \
        .negsamplegen(NegIdxGen(*negidxgenargs[0], **negidxgenargs[1])).negrate(negrate).objective(obj) \
        .adagrad(lr=lr).l2(wreg).grad_total_norm(1.0) \
        .validate_on([validdata, validgoldshifted, validgold]) \
        .train(numbats=numbats, epochs=epochs)
    tt.tock("trained")

    #scorer.save("scorer.test.save")

    # eval
    tt.tick("evaluating")
    s = SeqEncDecRankSearch(encdec, entenc, scorer.s, scorer.agg)
    eval = FullRankEval()
    pred, scores = s.decode(testdata, 0, testgold.shape[1],
                            candata=entmat, canids=canids,
                            transform=transf.f, debug=printpreds)
    if printpreds:
        print pred
    debugarg = "subj" if subjpred else "pred" if predpred else False
    evalres = eval.eval(pred, testgold, debug=debugarg)
    for k, evalre in evalres.items():
        print("{}:\t{}".format(k, evalre))
    tt.tock("evaluated")

    # save
    basename = os.path.splitext(os.path.basename(__file__))[0]
    dirname = basename + ".results"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    savenamegen = lambda i: "{}/{}.res".format(dirname, i)
    savename = None
    for i in xrange(100):
        savename = savenamegen(i)
        if not os.path.exists(savename):
            break
        savename = None
    if savename is None:
        raise Exception("exceeded number of saved results")
    with open(savename, "w") as f:
        f.write("{}\n".format(" ".join(sys.argv)))
        for k, evalre in evalres.items():
            f.write("{}:\t{}\n".format(k, evalre))

    #scorer.save(filepath=savename)


if __name__ == "__main__":
    argprun(run, debug=True)
