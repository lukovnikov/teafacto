from teafacto.util import ticktock, argprun, inp
import os, pickle, random
from teafacto.procutil import *
from IPython import embed
from scipy import sparse

from teafacto.blocks.lang.wordvec import Glove, WordEmb
from teafacto.blocks.lang.sentenc import TwoLevelEncoder
from teafacto.blocks.seq.rnn import RNNSeqEncoder, MaskMode
from teafacto.blocks.seq.enc import SimpleSeq2Vec
from teafacto.blocks.cnn import CNNSeqEncoder
from teafacto.blocks.basic import VectorEmbed
from teafacto.blocks.memory import MemVec
from teafacto.blocks.match import SeqMatchScore, CosineDistance, MatchScore

from teafacto.core.base import Block, tensorops as T, Val


def readdata(p="../../../../data/simplequestions/clean/datamat.word.fb2m.pkl",
             entinfp="../../../../data/simplequestions/clean/subjs-counts-labels-types.fb2m.tsv",
             cachep=None, #"subjpredcharns.readdata.cache.pkl",
             maskid=-1,
             debug=False,
             numtestcans=None,
             ):
    tt = ticktock("dataloader")
    if cachep is not None and os.path.isfile(cachep):      # load
        tt.tick("loading from cache")
        ret = pickle.load(open(cachep))
        tt.tock("loaded from cache")
    else:
        tt.tick("loading datamat")
        x = pickle.load(open(p))
        tt.tock("datamat loaded")
        worddic = x["worddic"]
        entdic = x["entdic"]
        entmat = x["entmat"]
        numents = x["numents"]
        traindata, traingold = x["train"]
        validdata, validgold = x["valid"]
        testdata, testgold = x["test"]
        traingold[:, 1] -= numents
        validgold[:, 1] -= numents
        testgold[:, 1] -= numents

        rwd = {v: k for k, v in worddic.items()}

        subjdic = {k: v for k, v in entdic.items() if v < numents}
        reldic = {k: v - numents for k, v in entdic.items() if v >= numents}

        subjmat = entmat[:numents]
        ssubjmat = np.sum(subjmat != maskid, axis=1)
        if np.any(ssubjmat == 0):
            for i in list(np.argwhere(ssubjmat == 0)[:, 0]):
                subjmat[i, 0] = worddic["<RARE>"]

        relmat = entmat[numents:]
        if debug:
            embed()

        traindata = wordmat2wordchartensor(traindata, rwd=rwd, maskid=maskid)
        validdata = wordmat2wordchartensor(validdata, rwd=rwd, maskid=maskid)
        testdata = wordmat2wordchartensor(testdata, rwd=rwd, maskid=maskid)

        subjmat = wordmat2charmat(subjmat, rwd=rwd, maskid=maskid, raretoken="<RARE>", maxlen=75)
        ret = ((traindata, traingold), (validdata, validgold),
               (testdata, testgold), (subjmat, relmat), (subjdic, reldic),
               worddic)
        if cachep is not None:
            tt.tick("dumping to cache")
            pickle.dump(ret, open(cachep, "w"))
            tt.tock("dumped to cache")

    subjinfo = loadsubjinfo(entinfp, subjdic)
    testsubjcans = loadsubjtestcans(numcans=numtestcans)
    testrelcans, relspersubj = loadreltestcans(testgold, subjdic, reldic)
    if debug:
        embed()
    return ret + (subjinfo, (testsubjcans, relspersubj))


def loadreltestcans(testgold, subjdic, reldic, relsperentp="../../../../data/simplequestions/allrelsperent.dmp"):
    tt = ticktock("test rel can loader")
    testsubjs = testgold[:, 0]
    relsperent = {} #{k: ([], []) for k in set(list(testsubjs))}
    tt.tick("loading rel test cans")
    for line in open(relsperentp):
        subj, relsout, relsin = line[:-1].split("\t")
        if subj in subjdic:
            relsperent[subjdic[subj]] = (
                    [reldic[x] for x in relsout.split(" ")] if relsout != "" else [],
                    [reldic[x] for x in relsin.split(" ")] if relsin != "" else []
            )
        #if subj in subjdic and subjdic[subj] in relsoftestsubjs:
        #    relsoftestsubjs[subjdic[subj]] = (
        #        [reldic[x] for x in relsout.split(" ")] if relsout != "" else [],
        #        [reldic[x] for x in relsin.split(" ")] if relsin != "" else []
        #    )
    tt.tock("test cans loaded")
    relsoftestexamples = [(relsperent[x][0], relsperent[x][1])
                          for x in testsubjs]
    return relsoftestexamples, relsperent


def loadsubjtestcans(p="../../../../data/simplequestions/clean/testcans{}.pkl", numcans=None):
    tt = ticktock("test subjects candidate loader")
    tt.tick("loading candidates")
    p = p.format("{}c".format(numcans)) if numcans is not None else p.format("")
    ret = pickle.load(open(p))
    tt.tock("canddiates loaded")
    return ret


def loadsubjinfo(entinfp, entdic, cachep=None):#"subjinfo.cache.pkl"):
    tt = ticktock("subjinfoloader")
    def make():
        tt.tick("making subject info from file")
        subjinf = {}
        c = 0
        for line in open(entinfp):
            subjuri, subjc, objc, subjname, typuri, typname = line[:-1].split("\t")
            subjinf[entdic[subjuri]] = (subjname, typname.lower().split(), typuri, subjc, objc)
            if c % 1000 == 0:
                tt.live(str(c))
            c += 1
        tt.tock("made subject info from file")
        return subjinf
    if cachep is not None:
        if os.path.isfile(cachep):      # load
            tt.tick("loading cached subject info")
            subjinfo = pickle.load(open(cachep))
            tt.tock("loaded cached subject info")
        else:                           # make  and dump
            subjinfo = make()
            tt.tick("dumping subject info in cache")
            pickle.dump(subjinfo, open(cachep, "w"))
            tt.tock("dumped subject info in cache")
    else:       # just make
        subjinfo = make()
    return subjinfo


def buildrelsamplespace(entmat, wd, maskid=-1):
    tt = ticktock("samplespace")
    tt.tick("making sample space")
    #rwd = {v: k for k, v in wd.items()}
    entmatm = sparse.dok_matrix((entmat.shape[0], np.max(entmat) + 1))
    posblacklist = {0: {wd["base"], wd["user"]}}
    blacklist = set([wd[x] for x in "default domain of by the in at s this for with type".split()])
    #revin = {k: set() for k in np.unique(entmat)}
    #revinm = sparse.dok_matrix((np.max(entmat), entmat.shape[0]))
    samdic = {k: set() for k in range(entmat.shape[0])}     # from ent ids to sets of ent ids
    #samdic = np.zeros((entmat.shape[0], entmat.shape[0]))
    for i in range(entmat.shape[0]):
        for j in range(entmat.shape[1]):
            w = entmat[i, j]
            if w == -1:     # beginning of padding
                break
            if j in posblacklist:
                if w in posblacklist[j]:
                    continue
            if w in blacklist:
                continue
            entmatm[i, w] = 1
            #for oe in revin[w]:     # other entities already in revind
            #    samdic[oe].add(i)
            #    samdic[i].add(oe)
            #revin[w].add(i)
            #revinm[w, i] = 1
    samdicm = entmatm.dot(entmatm.T)
    for i in range(samdicm.shape[0]):
        samdic[i] = list(np.argwhere(samdicm[i, :])[:, 1])
    tt.tock("made sample space")
    return samdic, entmatm.T


def buildtypmat(subjmat, subjinfo, worddic, maxlen=6, maskid=-1):
    ret = maskid * np.ones((subjmat.shape[0], maxlen), dtype="int32")
    import re
    splitterregex = re.compile("[\s/]")
    for i in range(ret.shape[0]):
        typstring = " ".join(subjinfo[i][1] if i in subjinfo else "<unk>")
        typwords = splitterregex.split(typstring)
        typwordids = [worddic[typword] if typword in worddic else 0 for typword in typwords]
        typwordids = typwordids[:min(len(typwords), maxlen)]
        ret[i, :len(typwordids)] = typwordids
    return ret


class SeqLeftBlock(Block):
    def __init__(self, inner, **kw):
        super(SeqLeftBlock, self).__init__(**kw)
        self.inner = inner

    def apply(self, x):
        # idxs^(batsize, seqlen, ...) --> (batsize, seqlen, 2, encdim)
        res = self.inner(x).dimshuffle(0, "x", 1)
        ret = T.concatenate([res, res], axis=1)
        return ret


class ConcatLeftBlock(Block):
    def __init__(self, inner, mid, **kw):
        super(ConcatLeftBlock, self).__init__(**kw)
        self.inner = inner
        self.mid = mid

    def apply(self, x):
        res = self.inner(x).dimshuffle(0, "x", 1) # (batsize, 1, q_enc_dim)
        mid = self.mid #res.shape[2]/2
        ret = T.concatenate([res[:, :, :mid], res[:, :, mid:]], axis=1)
        return ret


class RightBlock(Block):
    def __init__(self, a, b, **kw):
        super(RightBlock, self).__init__(**kw)
        self.subjenc = a
        self.predenc = b

    def apply(self, subjslice, relslice):  # idxs^(batsize, len)
        aret = self.subjenc(subjslice).dimshuffle(0, "x", 1)
        bret = self.predenc(relslice).dimshuffle(0, "x", 1)
        ret = T.concatenate([aret, bret], axis=1)
        return ret


class TypedSubjBlock(Block):
    def __init__(self, typelen, subjenc, typenc, **kw):
        super(TypedSubjBlock, self).__init__(**kw)
        self.typelen = typelen
        self.typenc = typenc
        self.subjenc = subjenc

    def apply(self, x):
        typewords = x[:, :self.typelen]
        subjchars = x[:, self.typelen:]
        typemb = self.typenc(typewords)
        subemb = self.subjenc(subjchars)
        ret = T.concatenate([subemb, typemb], axis=1)
        return ret


class CustomPredictor(object):
    def __init__(self, questionencoder=None, entityencoder=None,
                 relationencoder=None, mode=None,
                 enttrans=None, reltrans=None, debug=False,
                 subjinfo=None):
        self.qenc = questionencoder
        self.eenc = entityencoder
        self.renc = relationencoder
        self.mode = mode
        self.enttrans = enttrans
        self.reltrans = reltrans
        self.debug = debug
        self.subjinfo = subjinfo
        self.qencodings = None
        self.tt = ticktock("predictor")

    # stateful API
    def encodequestions(self, data):
        self.tt.tick("encoding questions")
        self.qencodings = self.qenc.predict(data)
        self.tt.tock("encoded questions")

    def ranksubjects(self, entcans):
        assert(self.qencodings is not None)
        if self.mode == "concat":
            mid = self.qencodings.shape[1] / 2
        elif self.mode == "seq":
            mid = self.qencodings.shape[1]
        else:
            raise Exception("unrecognized mode in prediction")
        qencforent = self.qencodings[:, :mid]
        self.tt.tick("rank subjects")
        ret = []    # list of lists of (subj, score) tuples, sorted
        for i in range(self.qencodings.shape[0]):       # for every question
            if len(entcans[i]) == 0:
                scoredentcans = [(-1, 0)]
            elif len(entcans[i]) == 1:
                scoredentcans = [(entcans[i][0], 1)]
            else:
                entembs = self.eenc.predict.transform(self.enttrans)(entcans[i])
                #embed()
                entscoresi = np.tensordot(qencforent[i], entembs, axes=(0, 1))
                entscoresi /= np.linalg.norm(qencforent[i])
                entscoresi /= np.linalg.norm(entembs, axis=1)
                scoredentcans = sorted(zip(entcans[i], entscoresi), key=lambda (x, y): y, reverse=True)
            ret.append(scoredentcans)
            self.tt.progress(i, self.qencodings.shape[0], live=True)
        self.tt.tock("ranked subjects")
        self.subjranks = ret
        return ret

    def rankrelations(self, relcans):
        assert(self.qencodings is not None)
        if self.mode == "concat":
            mid = self.qencodings.shape[1] / 2
        elif self.mode == "seq":
            mid = 0
        else:
            raise Exception("unrecognized mode in prediction")
        qencforrel = self.qencodings[:, mid:]
        self.tt.tick("rank relations")
        ret = []
        for i in range(self.qencodings.shape[0]):
            if len(relcans[i]) == 0:
                scoredrelcans = [(-1, 0)]
            elif len(relcans[i]) == 1:
                scoredrelcans = [(relcans[i][0], 1)]
            else:
                relembs = self.renc.predict.transform(self.reltrans)(relcans[i])
                relscoresi = np.tensordot(qencforrel[i], relembs, axes=(0, 1))
                relscoresi /= np.linalg.norm(qencforrel[i])
                relscoresi /= np.linalg.norm(relembs, axis=1)
                scoredrelcans = sorted(zip(relcans[i], relscoresi), key=lambda (x, y): y, reverse=True)
            ret.append(scoredrelcans)
            self.tt.progress(i, self.qencodings.shape[0], live=True)
        self.tt.tock("ranked relations")
        self.relranks = ret
        return ret

    def rankrelationsfroments(self, bestsubjs, relsperent):
        relcans = [relsperent[bestsubj][0] if bestsubj in relsperent else [] for bestsubj in bestsubjs]
        return self.rankrelations(relcans)

    def predict(self, data, entcans=None, relsperent=None, relcans=None):
        assert(relsperent is None or relcans is None)
        assert(relsperent is not None or relcans is not None)
        assert(entcans is not None)
        self.encodequestions(data)
        rankedsubjs = self.ranksubjects(entcans)
        bestsubjs = [x[0][0] for x in rankedsubjs]
        if relcans is not None:
            rankedrels = self.rankrelations(relcans)
        else:
            rankedrels = self.rankrelationsfroments(bestsubjs, relsperent)
        bestrels = [x[0][0] for x in rankedrels]

        ret = np.concatenate([
            np.expand_dims(np.asarray(bestsubjs, dtype="int32"), axis=1),
            np.expand_dims(np.asarray(bestrels, dtype="int32"), axis=1)
        ], axis=1)
        return ret

    def oldpredict(self, data, entcans, relsperent):
        tt = ticktock("predictor")
        tt.tick("computing question encodings")
        qencodings = self.qenc.predict(data)    # (numsam, encdim)
        tt.tock("computed question encodings")
        tt.tick("predicting")
        ret = np.zeros((data.shape[0], 2), dtype="int32")
        if self.mode == "concat":
            mid = qencodings.shape[1] / 2
            qencforent = qencodings[:, :mid]
            qencforrel = qencodings[:, mid:]
        elif self.mode == "seq":
            qencforent = qencodings[:, :]
            qencforrel = qencodings[:, :]
        else:
            raise Exception("unrecognized mode")
        for i in range(qencodings.shape[0]):
            # predict subject
            if len(entcans[i]) == 0:
                bestsubj = -1
            elif len(entcans[i]) == 1:
                bestsubj = entcans[i][0]
            else:
                entembs = self.eenc.predict.transform(self.enttrans)(entcans[i])
                entscoresi = np.tensordot(qencforent[i], entembs, axes=(0, 1))
                scoredentcans = sorted(zip(entcans[i], entscoresi), key=lambda (x, y): y, reverse=True)
                bestsubj = scoredentcans[0][0]
                if self.debug:
                    embed()
            ret[i, 0] = bestsubj
            # predict relation
            relcans = relsperent[ret[i, 0]][0] if ret[i, 0] in relsperent else []
            if len(relcans) == 0:
                bestrel = -1
            elif len(relcans) == 1:
                bestrel = relcans[0]
            else:
                if self.debug:
                    embed()
                relembs = self.renc.predict.transform(self.reltrans)(relcans)
                relscoresi = np.tensordot(qencforrel[i], relembs, axes=(0, 1))
                scoredrelcans = sorted(zip(relcans, relscoresi), key=lambda (x, y): y, reverse=True)
                bestrel = scoredrelcans[0][0]
            ret[i, 1] = bestrel
            if self.debug:
                embed()
            tt.progress(i, qencodings.shape[0], live=True)
        tt.tock("predicted")
        return ret


def run(closenegsam=False,
        checkdata=False,
        glove=True,
        embdim=100,
        charencdim=100,
        charembdim=50,
        encdim=400,
        bidir=False,
        charenc="cnn",  # "cnn" or "rnn"
        mode="seq",      # "seq" or "concat"
        margin=0.5,
        lr=0.1,
        numbats=700,
        epochs=15,
        debug=False,
        gradnorm=1.0,
        wreg=0.0001,
        loadmodel=-1,
        debugtest=False,
        forcesubjincl=False,
        usetypes=False,
        randsameval=0,
        numtestcans=0,
        ):
    tt = ticktock("script")
    tt.tick("loading data")
    (traindata, traingold), (validdata, validgold), (testdata, testgold), \
    (subjmat, relmat), (subjdic, reldic), worddic, \
    subjinfo, (testsubjcans, relsperent) = readdata(debug=debug,
                                                    numtestcans=numtestcans if numtestcans > 0 else None)

    if usetypes:
        print "building type matrix"
        typmat = buildtypmat(subjmat, subjinfo, worddic)
        subjmat = np.concatenate([typmat, subjmat], axis=1)
        typlen = typmat.shape[1]

    relsamplespace = None
    if closenegsam:
        relsamplespace, revind = buildrelsamplespace(relmat, worddic)
    tt.tock("data loaded")

    if checkdata:
        embed()

    numwords = max(worddic.values()) + 1
    numsubjs = max(subjdic.values()) + 1
    numrels = max(reldic.values()) + 1
    maskid = -1
    numchars = 256

    if mode == "seq":
        decdim = encdim
        print "seq mode: {} decdim".format(decdim)
    elif mode == "concat":
        decdim = encdim / 2
        print "concat mode: {} decdim".format(decdim)
    else:
        raise Exception("unrecognized mode")

    # defining model
    if glove:
        wordemb = Glove(embdim).adapt(worddic)
    else:
        wordemb = WordEmb(dim=embdim, indim=numwords)
    charemb = VectorEmbed(indim=numchars, dim=charembdim)
    if charenc == "cnn":
        print "using CNN char encoder"
        charenc = CNNSeqEncoder(inpemb=charemb,
                                innerdim=[charencdim]*2, maskid=maskid,
                                stride=1)
    elif charenc == "rnn":
        print "using RNN char encoder"
        charenc = RNNSeqEncoder(inpemb=charemb, innerdim=charencdim)\
            .maskoptions(maskid, MaskMode.AUTO)
    else:
        raise Exception("no other modes available")
    wordenc = RNNSeqEncoder(inpemb=False, inpembdim=wordemb.outdim + charencdim,
                            innerdim=encdim, bidir=bidir).maskoptions(MaskMode.NONE)
    question_encoder = TwoLevelEncoder(l1enc=charenc, l2emb=wordemb,
                                       l2enc=wordenc, maskid=maskid)

    # encode predicate on word level
    predemb = SimpleSeq2Vec(inpemb=wordemb,
                           innerdim=decdim,
                           maskid=maskid,
                           bidir=bidir,
                           layers=1)
    #predemb.load(relmat)

    if usetypes:
        # encode subj type on word level
        subjtypemb = SimpleSeq2Vec(inpemb=wordemb,
                                   innerdim=int(np.ceil(decdim*1./3)),
                                   maskid=maskid,
                                   bidir=bidir,
                                   layers=1)
        # encode subject on character level
        subjemb = SimpleSeq2Vec(inpemb=charemb,
                               innerdim=int(np.floor(decdim*2./3)),
                               maskid=maskid,
                               bidir=bidir,
                               layers=1)
        subjemb = TypedSubjBlock(typlen, subjemb, subjtypemb)
    else:
        # encode subject on character level
        subjemb = SimpleSeq2Vec(inpemb=charemb,
                               innerdim=decdim,
                               maskid=maskid,
                               bidir=bidir,
                               layers=1)
    #subjemb.load(subjmat)

    # package
    if mode == "seq":
        lb = SeqLeftBlock(question_encoder)
        rb = RightBlock(subjemb, predemb)
    elif mode == "concat":
        lb = ConcatLeftBlock(question_encoder, decdim)
        rb = RightBlock(subjemb, predemb)
    else:
        raise Exception("unrecognized mode")
    scorer = SeqMatchScore(lb, rb, scorer=CosineDistance(),
                           aggregator=lambda x: x, argproc=lambda x, y, z: ((x,), (y, z)))

    obj = lambda p, n: T.sum((n - p + margin).clip(0, np.infty), axis=1)

    # negative sampling
    class NegIdxGen(object):
        def __init__(self, maxentid, maxrelid, relclose=None):
            self.maxentid = maxentid
            self.maxrelid = maxrelid
            self.relclose = relclose

        def __call__(self, datas, gold):
            entrand = np.random.randint(0, self.maxentid+1, (gold.shape[0], 1))

            #if np.sum(entrand == 645994) > 0:
            #    print "sampled the empty subject label"
            #entrand = np.vectorize(lambda x: 645995 if x == 645994 else x)(entrand) # avoid sampling the empty label
            #entrand = np.asarray([[645994]]*gold.shape[0])
            #entrand[0, 0] = 645994

            relrand = self.samplerels(gold[:, 1])
            ret = np.concatenate([entrand, np.expand_dims(relrand, axis=1)], axis=1)
            #embed()
            return datas, ret.astype("int32")

        def samplerels(self, gold):
            #assert(gold.ndim == 2 and gold.shape[1] == 1)
            if self.relclose is None:
                return np.random.randint(0, self.maxrelid+1, (gold.shape[0], 1))
            else:
                ret = np.zeros_like(gold)
                for i in range(gold.shape[0]):
                    sampleset = self.relclose[gold[i]]
                    if len(sampleset) > 5:
                        ret[i] = random.sample(sampleset, 1)[0]
                    else:
                        ret[i] = np.random.randint(0, self.maxrelid+1)
                return ret.astype("int32")

    class PreProc(object):
        def __init__(self, subjmat, relmat):
            self.ef = PreProcEnt(subjmat)
            self.rf = PreProcEnt(relmat)

        def __call__(self, data, gold):     # gold: idxs-(batsize, 2)
            st = self.ef(gold[:, 0])[0][0]
            rt = self.rf(gold[:, 1])[0][0]
            return (data, st, rt), {}

    class PreProcE(object):
        def __init__(self, subjmat, relmat):
            self.ef = PreProcEnt(subjmat)
            self.rf = PreProcEnt(relmat)

        def __call__(self, x):
            subjslice = self.ef(x[:, 0])[0][0]
            relslice = self.rf(x[:, 1])[0][0]
            return (subjslice, relslice), {}

    class PreProcEnt(object):
        def __init__(self, mat):
            self.entmat = Val(mat)

        def __call__(self, x):
            return (self.entmat[x],), {}

    transf = PreProc(subjmat, relmat)

    if debug:
        embed()

    if epochs > 0 and loadmodel < 0:
        tt.tick("training")
        nscorer = scorer.nstrain([traindata, traingold]).transform(transf)\
            .negsamplegen(NegIdxGen(numsubjs-1, numrels-1, relclose=relsamplespace)) \
            .objective(obj).adagrad(lr=lr).l2(wreg).grad_total_norm(gradnorm)\
            .validate_on([validdata, validgold])\
            .train(numbats=numbats, epochs=epochs)
        tt.tock("trained").tick()

        # saving
        saveid = "".join([str(np.random.randint(0, 10)) for i in range(4)])
        scorer.save("fullrank{}.model".format(saveid))
        tt.tock("saved: {}".format(saveid))

    if loadmodel > -1:
        tt.tick("loading model")
        m = SeqMatchScore.load("fullrank{}.model".format(loadmodel))
        #embed()
        question_encoder = m.l.inner
        subjemb = m.r.subjenc
        predemb = m.r.predenc
        tt.tock("loaded model")

    # evaluation
    predictor = CustomPredictor(questionencoder=question_encoder,
                                entityencoder=subjemb,
                                relationencoder=predemb,
                                mode=mode,
                                enttrans=transf.ef,
                                reltrans=transf.rf,
                                debug=debugtest,
                                subjinfo=subjinfo)

    tt.tick("predicting")
    if forcesubjincl:       # forces the intended subject entity to be among candidates
        for i in range(len(testsubjcans)):
            if testgold[i, 0] not in testsubjcans[i]:
                testsubjcans[i].append(testgold[i, 0])

    if randsameval > 0:     # generate random sampling eval data
        testsubjcans = np.random.randint(0, numsubjs, (testgold.shape[0], randsameval))
        testrelcans = np.random.randint(0, numrels, (testgold.shape[0], randsameval))
        testsubjcans = np.concatenate([testgold[:, 0:1], testsubjcans], axis=1)
        testrelcans = np.concatenate([testgold[:, 1:2], testrelcans], axis=1)
        testsubjcans = testsubjcans.tolist()
        testrelcans = testrelcans.tolist()
        prediction = predictor.predict(testdata, entcans=testsubjcans, relcans=testrelcans)
    else:
        prediction = predictor.predict(testdata, entcans=testsubjcans, relsperent=relsperent)
    tt.tock("predicted")
    tt.tick("evaluating")
    evalmat = prediction == testgold
    subjacc = np.sum(evalmat[:, 0]) * 1. / evalmat.shape[0]
    predacc = np.sum(evalmat[:, 1]) * 1. / evalmat.shape[0]
    totalacc = np.sum(np.sum(evalmat, axis=1) == 2) * 1. / evalmat.shape[0]
    print "Test results ::::::::::::::::"
    print "Total Acc: \t {}".format(totalacc)
    print "Subj Acc: \t {}".format(subjacc)
    print "Pred Acc: \t {}".format(predacc)
    tt.tock("evaluated")

    def subjinspect(subjrank, gold):
        ret = [(("GOLD - " if gold == x else "       ") +
                subjinfo[x][0] + " (" + " ".join(subjinfo[x][1]) + ")" +
                str(subjinfo[x][3]) + " rels",
                y) if x in subjinfo else (x, y)
               for x, y in subjrank]
        return ret

    def inspectsubjs():
        rwd = {v: k for k, v in worddic.items()}
        for i in range(len(predictor.subjranks)):
            subjx = testgold[i, 0]

            print "test question {}: {} \t GOLD: {}".format(i,
                                                wordids2string(testdata[i, :, 0], rwd),
                                                "{} ({}) - {} rels".format(
                                                    *([subjinfo[subjx][0],
                                                    subjinfo[subjx][1],
                                                    subjinfo[subjx][3]]
                                                    if subjx in subjinfo
                                                    else ["<UNK>", "<UNK>", "<UNK>"])
                                                ))
            subjrank = predictor.subjranks[i]
            gold = testgold[i, 0]
            inspres = subjinspect(subjrank, gold)
            for inspre in inspres:
                print inspre
            inp()

    embed()




if __name__ == "__main__":
    argprun(run)