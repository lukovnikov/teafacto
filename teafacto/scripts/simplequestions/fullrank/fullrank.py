from teafacto.util import ticktock, argprun, inp, tokenize
import os, pickle, random
from teafacto.procutil import *
from IPython import embed
from scipy import sparse

from teafacto.blocks.lang.wordvec import Glove, WordEmb
from teafacto.blocks.lang.sentenc import TwoLevelEncoder
from teafacto.blocks.seq.rnn import RNNSeqEncoder, MaskMode
from teafacto.blocks.seq.enc import SimpleSeq2Vec, SimpleSeq2MultiVec, SimpleSeq2Sca, EncLastDim
from teafacto.blocks.cnn import CNNSeqEncoder
from teafacto.blocks.basic import VectorEmbed, MatDot, Linear
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


def loadsubjsamplespace(p="../../../../data/simplequestions/clean/subjclose.dic.pkl"):
    d = pickle.load(open(p))
    return d

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
        return ret      # (batsize, 2, decdim)


class ConcatLeftBlock(Block):
    def __init__(self, inner, **kw):
        super(ConcatLeftBlock, self).__init__(**kw)
        self.trans = MatDot(inner.outdim, inner.outdim, init="glorotuniform") \
            if inner.bidir else lambda x: x
        self.inner = inner

    def apply(self, x):
        res = self.inner(x)
        res = self.trans(res)
        res = res.dimshuffle(0, "x", 1)     # (batsize, 1, q_enc_dim)
        if not self.inner.bidir:
            mid = res.shape[2]/2
            ret = T.concatenate([res[:, :, :mid], res[:, :, mid:]], axis=1)
        else:
            quart = res.shape[2]/2
            ret = T.concatenate([
                T.concatenate([res[:, :, :quart], res[:, :, 2*quart:3*quart]], axis=2),
                T.concatenate([res[:, :, quart:2*quart], res[:, :, 3*quart:]], axis=2)
            ], axis=1)
        return ret      # (batsize, 2, decdim)


class MultiLeftBlock(Block):
    def __init__(self, inner, mode, **kw):
        super(MultiLeftBlock, self).__init__(**kw)
        self.inner = inner
        self.mode = mode

    def apply(self, x):
        res = self.inner(x)             # (batsize, 2, encdim)
        if self.mode == "multic":   # take top half of first and bottom half of second
            if not self.inner.bidir:
                mid = res.shape[2]/2
                ret = T.concatenate([res[:, 0:1, :mid], res[:, 1:2, mid:]], axis=1)
            else:
                quarts = res.shape[2]/4
                ret = T.concatenate([
                            T.concatenate([ res[:, 0:1, :quarts],
                                            res[:, 0:1, 2*quarts:3*quarts]], axis=2),
                            T.concatenate([ res[:, 1:2, quarts:2*quarts],
                                            res[:, 1:2, 3*quarts:]], axis=2)
                ], axis=1)
        else:                       # return as is
            ret = res
        print "NDIM MULTILEFTBLOCK !!!!!!!!!!!!!!!!!!!!!{}".format(ret.ndim)
        return ret      # (batsize, 2, decdim)


class BinoEncoder(Block):
    def __init__(self, charenc=None, wordemb=None, maskid=-1, scalayers=1,
            scadim=100, encdim=100, outdim=None, scabidir=False, encbidir=False, enclayers=1, **kw):
        super(BinoEncoder, self).__init__(**kw)
        self.charenc = charenc
        self.wordemb = wordemb
        self.maskid = maskid
        self.bidir = encbidir  # TODO
        outdim = encdim if outdim is None else outdim
        self.outdim = outdim  # TODO
        self.outerpol = SimpleSeq2Sca(inpemb=False, inpembdim=charenc.outdim + wordemb.outdim,
                                      innerdim=[scadim]*scalayers, bidir=scabidir)
        self.leftenc = RNNSeqEncoder(inpemb=False, inpembdim=charenc.outdim + wordemb.outdim,
                                     innerdim=[encdim]*enclayers, bidir=encbidir, maskid=maskid)
        self.rightenc = RNNSeqEncoder(inpemb=False, inpembdim=charenc.outdim + wordemb.outdim,
                                      innerdim=[encdim]*enclayers, bidir=encbidir, maskid=maskid)
        self.leftlin = Linear(self.leftenc.outdim, outdim)
        self.rightlin = Linear(self.rightenc.outdim, outdim)

    def apply(self, x):
        # word vectors and mask
        charten = x[:, :, 1:]
        charencs = EncLastDim(self.charenc)(charten)
        wordmat = x[:, :, 0]
        wordembs = self.wordemb(wordmat)
        wordvecs = T.concatenate([charencs, wordembs], axis=2)
        wordmask = T.neq(wordmat, self.maskid)
        wordvecs.mask = wordmask
        # do outerpolation
        weights, mask = self.outerpol(wordvecs)
        leftenco = self.leftenc(wordvecs, weights=weights).dimshuffle(0, 'x', 1)
        rightenco = self.rightenc(wordvecs, weights=(1 - weights)).dimshuffle(0, 'x', 1)
        ret = T.concatenate([self.leftlin(leftenco),
                             self.rightlin(rightenco)],
                            axis=1)
        return ret      # (batsize, 2, decdim)





class RightBlock(Block):
    def __init__(self, a, b, **kw):
        super(RightBlock, self).__init__(**kw)
        self.subjenc = a
        self.predenc = b

    def apply(self, subjslice, relslice):  # idxs^(batsize, len)
        aret = self.subjenc(subjslice).dimshuffle(0, "x", 1)
        bret = self.predenc(relslice).dimshuffle(0, "x", 1)
        ret = T.concatenate([aret, bret], axis=1)
        return ret  # (batsize, 2, decdim)


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
                 relationencoder=None,
                 enttrans=None, reltrans=None, debug=False,
                 subjinfo=None):
        self.qenc = questionencoder
        self.eenc = entityencoder
        self.renc = relationencoder
        #self.mode = mode
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
        qencforent = self.qencodings[:, 0, :]
        '''if self.mode == "concat":
            qencforent = self.qencodings[:, :(self.qencodings.shape[1] / 2)]
        elif self.mode == "seq":
            qencforent = self.qencodings[:, :]
        elif self.mode == "multi":
            qencforent = self.qencodings[:, 0, :]
        elif self.mode == "multic":
            qencforent = self.qencodings[:, 0, :(self.qencodings.shape[2] / 2)]
        else:
            raise Exception("unrecognized mode in prediction")'''
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
        qencforrel = self.qencodings[:, 1, :]
        '''if self.mode == "concat":
            qencforrel = self.qencodings[:, (self.qencodings.shape[1] / 2):]
        elif self.mode == "seq":
            qencforrel = self.qencodings[:, :]
        elif self.mode == "multi":
            qencforrel = self.qencodings[:, 1, :]
        elif self.mode == "multic":
            qencforrel = self.qencodings[:, 1, (self.qencodings.shape[2] / 2):]
        else:
            raise Exception("unrecognized mode in prediction")'''
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

    def predict(self, data, entcans=None, relsperent=None, relcans=None, multiprune=-1):
        print multiprune
        assert(relsperent is None or relcans is None)
        assert(relsperent is not None or relcans is not None)
        assert(entcans is not None)
        self.encodequestions(data)
        rankedsubjs = self.ranksubjects(entcans)
        bestsubjs = [x[0][0] for x in rankedsubjs]
        if relcans is not None:
            rankedrels = self.rankrelations(relcans)
            bestrels = [x[0][0] for x in rankedrels]
        else:
            if multiprune <= 0:
                relcans = [relsperent[bestsubj][0] if bestsubj in relsperent else [] for bestsubj in bestsubjs]
                rankedrels = self.rankrelations(relcans)
                bestrels = [x[0][0] for x in rankedrels]
            else:
                print "multipruning !!!!!!!!!!!!!!!!!"
                topk = multiprune        # TOP K !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # get relcans
                relcans = []
                for subjranking in rankedsubjs:
                    toplabel = None
                    relcanse = []
                    i = 0
                    for subj, score in subjranking:
                        subjlabel = " ".join(tokenize(self.subjinfo[subj][0]) if subj in self.subjinfo else [])
                        topcan = None
                        if toplabel is None:
                            toplabel = subjlabel
                            topcan = subj
                        elif subjlabel == toplabel:
                            topcan = subj
                        elif i < topk:
                            topcan = subj
                        else:
                            pass
                        toadd = relsperent[topcan][0] if topcan in relsperent else []
                        relcanse.extend(toadd)
                        i += 1
                    relcans.append(relcanse)
                # rank relations
                rankedrels = self.rankrelations(relcans)
                bestrels = [x[0][0] for x in rankedrels]
                # build ents per relation
                entsperrel = {}
                for ent, rels in relsperent.items():
                    for rel in rels[0]:
                        if rel not in entsperrel:
                            entsperrel[rel] = set()
                        entsperrel[rel].add(ent)
                # filter rankedsubjs
                filteredrankedsubjs = []
                for i in range(len(rankedsubjs)):
                    filteredrankedsubjs.append([])
                    for subj, score in rankedsubjs[i]:
                        if bestrels[i] in entsperrel and \
                                        subj in entsperrel[bestrels[i]]:
                                filteredrankedsubjs[i].append((subj, score))
                    if len(filteredrankedsubjs[i]) == 0:
                        filteredrankedsubjs[i].append((-1, -1.))
                bestsubjs = [x[0][0] for x in filteredrankedsubjs]






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


class NegIdxGen(object):
    def __init__(self, maxentid, maxrelid, relclose=None, subjclose=None, relsperent=None):
        self.maxentid = maxentid
        self.maxrelid = maxrelid
        print "using relclose" if relclose is not None else "no relclose"
        print "using subjclose" if subjclose is not None else "no subjclose"
        print "using relsperent" if relsperent is not None else "no relsperent"
        self.relclose = {k: set(v) for k, v in relclose.items()} if relclose is not None else None
        self.subjclose = {k: set(v) for k, v in subjclose.items()} if subjclose is not None else None
        self.relsperent = {k: set(v[0]) for k, v in relsperent.items()} if relsperent is not None else None
        self.samprobf = lambda x: np.tanh(np.log(x + 1)/3)

    def __call__(self, datas, gold):
        subjrand = self.sample(gold[:, 0], self.subjclose, self.maxentid)
        if self.relsperent is not None:     # sample uber-close
            relrand = self.samplereluberclose(gold[:, 1], gold[:, 0])
        else:
            relrand = self.sample(gold[:, 1], self.relclose, self.maxrelid)
        ret = np.concatenate([subjrand, relrand], axis=1)
        # embed()
        # TODO NEGATIVE SAMPLING OF RELATIONS FROM GOLD ENTITY'S RELATIONS
        return datas, ret.astype("int32")

    def samplereluberclose(self, relgold, entgold):
        ret = np.zeros_like(relgold, dtype="int32")
        for i in range(relgold.shape[0]):
            uberclosesampleset = (self.relsperent[entgold[i]] if entgold[i] in self.relsperent else set())\
                .difference({relgold[i]})
            if np.random.random() < self.samprobf(len(uberclosesampleset)):
                ret[i] = random.sample(uberclosesampleset, 1)[0]
            else:
                completerandom = False
                if self.relclose is not None:
                    closesampleset = (self.relclose[relgold[i]] if relgold[i] in self.relclose else set())\
                        .difference({relgold[i]})
                    if np.random.random() < self.samprobf(len(closesampleset)):
                        ret[i] = random.sample(closesampleset, 1)[0]
                    else:
                        completerandom = True
                else:
                    completerandom = True
                if completerandom:
                    ret[i] = np.random.randint(0, self.maxrelid + 1)
        ret = np.expand_dims(ret, axis=1)
        return ret

    def sample(self, gold, closeset, maxid):
        # assert(gold.ndim == 2 and gold.shape[1] == 1)
        if closeset is None:
            return np.random.randint(0, maxid + 1, (gold.shape[0], 1))
        else:
            ret = np.zeros_like(gold)
            for i in range(gold.shape[0]):
                sampleset = closeset[gold[i]] if gold[i] in closeset else []
                if np.random.random() < self.samprobf(len(sampleset)):
                    ret[i] = random.sample(sampleset, 1)[0]
                else:
                    ret[i] = np.random.randint(0, maxid + 1)
            ret = np.expand_dims(ret, axis=1)
            return ret.astype("int32")


def run(negsammode="closest",   # "close" or "random"
        usetypes=True,
        mode="concat",      # "seq" or "concat" or "multi" or "multic" or "bino"
        glove=True,
        embdim=100,
        charencdim=100,
        charembdim=50,
        encdim=400,
        bidir=False,
        layers=1,
        charenc="rnn",  # "cnn" or "rnn"
        margin=0.5,
        lr=0.1,
        numbats=700,
        epochs=15,
        gradnorm=1.0,
        wreg=0.0001,
        loadmodel="no",
        debug=False,
        debugtest=False,
        forcesubjincl=False,
        randsameval=0,
        numtestcans=5,
        multiprune=-1,
        checkdata=False,
        testnegsam=False,
        testmodel=False,
        sepcharembs=False,
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
    subjsamplespace = None
    if negsammode == "closest" or negsammode == "close":
        relsamplespace, revind = buildrelsamplespace(relmat, worddic)
        subjsamplespace = loadsubjsamplespace()
    tt.tock("data loaded")

    if checkdata:
        embed()

    numwords = max(worddic.values()) + 1
    numsubjs = max(subjdic.values()) + 1
    numrels = max(reldic.values()) + 1
    maskid = -1
    numchars = 256

    nsrelsperent = relsperent if negsammode == "closest" else None

    if testnegsam:
        nig = NegIdxGen(numsubjs - 1, numrels - 1,
                        relclose=relsamplespace,
                        subjclose=subjsamplespace,
                        relsperent=nsrelsperent)
        embed()

    if mode == "seq" or mode == "multi":
        decdim = encdim
    elif mode == "concat" or mode == "multic" or mode == "bino":
        decdim = encdim / 2
    else:
        raise Exception("unrecognized mode")

    print "{} mode: {} decdim".format(mode, decdim)

    # defining model
    if glove:
        wordemb = Glove(embdim).adapt(worddic)
    else:
        wordemb = WordEmb(dim=embdim, indim=numwords)

    charemb = VectorEmbed(indim=numchars, dim=charembdim)
    charemb2 = VectorEmbed(indim=numchars, dim=charembdim)
    if charenc == "cnn":
        print "using CNN char encoder"
        charenc = CNNSeqEncoder(inpemb=charemb,
                                innerdim=[charencdim]*2, maskid=maskid,
                                stride=1)
    elif charenc == "rnn":
        print "using RNN char encoder"
        charenc = RNNSeqEncoder(inpemb=charemb, innerdim=charencdim) \
            .maskoptions(maskid, MaskMode.AUTO)
    else:
        raise Exception("no other character encoding modes available")

    if bidir:
        encdim = encdim / 2

    if mode != "bino":
        if mode == "multi" or mode == "multic":
            wordenc = \
                SimpleSeq2MultiVec(inpemb=False, inpembdim=wordemb.outdim + charencdim,
                                   innerdim=encdim, bidir=bidir, numouts=2, mode="seq")
        else:
            encdim = [encdim] * layers
            wordenc = RNNSeqEncoder(inpemb=False, inpembdim=wordemb.outdim + charencdim,
                                    innerdim=encdim, bidir=bidir).maskoptions(MaskMode.NONE)

        question_encoder = TwoLevelEncoder(l1enc=charenc, l2emb=wordemb,
                                           l2enc=wordenc, maskid=maskid)

    else:
        question_encoder = BinoEncoder(charenc=charenc, wordemb=wordemb, maskid=maskid,
                                       scadim=100, encdim=encdim/2, bidir=bidir,
                                       enclayers=layers, outdim=decdim, scabidir=True)

    # encode predicate on word level
    predemb = SimpleSeq2Vec(inpemb=wordemb,
                            innerdim=decdim,
                            maskid=maskid,
                            bidir=False,
                            layers=1)

    #predemb.load(relmat)

    scharemb = charemb2 if sepcharembs else charemb
    if usetypes:
        # encode subj type on word level
        subjtypemb = SimpleSeq2Vec(inpemb=wordemb,
                                   innerdim=int(np.ceil(decdim*1./2)),
                                   maskid=maskid,
                                   bidir=False,
                                   layers=1)
        # encode subject on character level
        charbidir = True
        charencinnerdim = int(np.floor(decdim*1./2))
        charenclayers = 1
        if charbidir:
            charencinnerdim /= 2
            charenclayers = 2
        subjemb = SimpleSeq2Vec(inpemb=scharemb,
                                innerdim=charencinnerdim,
                                maskid=maskid,
                                bidir=charbidir,
                                layers=charenclayers)
        subjemb = TypedSubjBlock(typlen, subjemb, subjtypemb)
    else:
        # encode subject on character level
        subjemb = SimpleSeq2Vec(inpemb=scharemb,
                                innerdim=decdim,
                                maskid=maskid,
                                bidir=False,
                                layers=1)
    #subjemb.load(subjmat)
    if testmodel:
        embed()
    # package
    if mode == "seq":
        lb = SeqLeftBlock(question_encoder)
        rb = RightBlock(subjemb, predemb)
    elif mode == "concat":
        lb = ConcatLeftBlock(question_encoder)
        rb = RightBlock(subjemb, predemb)
    elif mode == "multi" or mode == "multic":
        lb = MultiLeftBlock(question_encoder, mode)
        rb = RightBlock(subjemb, predemb)
    elif mode == "bino":
        lb = question_encoder
        rb = RightBlock(subjemb, predemb)
    else:
        raise Exception("unrecognized mode")
    scorer = SeqMatchScore(lb, rb, scorer=CosineDistance(),
                           aggregator=lambda x: x, argproc=lambda x, y, z: ((x,), (y, z)))

    obj = lambda p, n: T.sum((n - p + margin).clip(0, np.infty), axis=1)

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

    if epochs > 0 and loadmodel == "no":
        tt.tick("training")
        saveid = "".join([str(np.random.randint(0, 10)) for i in range(4)])
        print("CHECKPOINTING AS: {}".format(saveid))
        nscorer = scorer.nstrain([traindata, traingold]).transform(transf) \
            .negsamplegen(NegIdxGen(numsubjs-1, numrels-1,
                                    relclose=relsamplespace,
                                    subjclose=subjsamplespace,
                                    relsperent=nsrelsperent)) \
            .objective(obj).adagrad(lr=lr).l2(wreg).grad_total_norm(gradnorm) \
            .validate_on([validdata, validgold]) \
            .autosavethis(scorer, "fullrank{}.model".format(saveid)) \
            .train(numbats=numbats, epochs=epochs)
        tt.tock("trained").tick()

        # saving
        #scorer.save("fullrank{}.model".format(saveid))
        print("SAVED AS: {}".format(saveid))

    if loadmodel is not "no":
        tt.tick("loading model")
        m = SeqMatchScore.load("fullrank{}.model".format(loadmodel))
        #embed()
        lb = m.l
        subjemb = m.r.subjenc
        predemb = m.r.predenc
        tt.tock("loaded model")

    # evaluation
    predictor = CustomPredictor(questionencoder=lb,
                                entityencoder=subjemb,
                                relationencoder=predemb,
                                #mode=mode,
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
        prediction = predictor.predict(testdata, entcans=testsubjcans,
                                       relsperent=relsperent, multiprune=multiprune)
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


    def inspectboth(hidecorrect=False, hidenotincan=False):
        rwd = {v: k for k, v in worddic.items()}
        for i in range(len(predictor.subjranks)):
            subjx = testgold[i, 0]
            predx = testgold[i, 1]
            subjrank = predictor.subjranks[i]
            predrank = predictor.relranks[i]
            if hidecorrect and subjx == subjrank[0][0] and predrank[0][0] == predx:
                continue
            if subjx not in [k for k, v in subjrank]:
                if hidenotincan:
                    continue



    def inspectsubjs(hidecorrect=False, hidenotincan=False, shownotincan=False):
        rwd = {v: k for k, v in worddic.items()}
        for i in range(len(predictor.subjranks)):
            subjx = testgold[i, 0]
            subjrank = predictor.subjranks[i]
            if subjx == subjrank[0][0] and hidecorrect:     # only look for errors
                continue
            if subjx not in [k for k, v in subjrank]:
                if hidenotincan:
                    continue
            if shownotincan and subjx in [k for k, v in subjrank]:
                continue
            print "test question {}: {} \t GOLD: {}".format(i,
                                                            wordids2string(testdata[i, :, 0], rwd),
                                                            "{} ({}) - {} rels --- {}".format(
                                                                *([subjinfo[subjx][0],
                                                                   subjinfo[subjx][1],
                                                                   subjinfo[subjx][3],
                                                                   subjinfo[subjx][2]]
                                                                  if subjx in subjinfo
                                                                  else ["<UNK>", "<UNK>", "<UNK>", "<UNK>"])
                                                            ))
            inspres = subjinspect(subjrank, subjx)
            i = 1
            for inspre in inspres:
                print "{}:\t{}\t{}".format(i, inspre[1], inspre[0])
                if i % 50 == 0:
                    inp()
                i += 1
            inp()

    def inspectpreds(hidecorrect=False):
        rwd = {v: k for k, v in worddic.items()}
        for i in range(len(predictor.relranks)):
            relx = testgold[i, 1]
            subjx = testgold[i, 0]
            relrank = predictor.relranks[i]
            if relx == relrank[0][0] and hidecorrect:
                continue
            print "test question {}: {} \t GOLD: {}".format(i,
                                                            wordids2string(testdata[i, :, 0], rwd),
                                                            wordids2string(relmat[relx, :], rwd))
            inspres = [(("GOLD - " if relx == x else "        ") +
                        wordids2string(relmat[x], rwd), y) for x, y in relrank]
            i = 1
            for inspre in inspres:
                print "{}:\t{}\t{}".format(i, inspre[1], inspre[0])
                if i % 50 == 0:
                    inp()
                i += 1
            inp()

    embed()




if __name__ == "__main__":
    argprun(run)