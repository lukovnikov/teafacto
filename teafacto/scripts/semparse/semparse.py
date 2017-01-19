from teafacto.util import argprun, isstring, issequence
from teafacto.procutil import wordids2string, wordmat2charmat
import numpy as np, re, random
from IPython import embed

from teafacto.core.base import Val, Block, tensorops as T
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt
from teafacto.blocks.basic import VectorEmbed, Linear, MatDot
from teafacto.blocks.activations import Softmax, Tanh
from teafacto.blocks.lang.wordvec import WordEmb, Glove
from teafacto.query.lbd2tree import LambdaParser


def loadgeopl(p="../../../data/semparse/geoquery.txt", customemb=False, reverse=True, splitre=None):
    qss, ass = [], []
    maxqlen, maxalen = 0, 0
    qwords, awords = {"<RARE>": 1}, {}

    if isstring(p):
        p = open(p)

    for line in p:
        splitre = ("[\s-]" if customemb else "\s") if splitre is None else splitre
        q, a = [re.split(splitre, x) for x in line.split("\t")]
        q = ["<s>"] + q + ["</s>"]
        a = ["<s>"] + a + ["</s>"]
        qss.append(q)
        ass.append(a)
        maxqlen = max(len(q), maxqlen)
        maxalen = max(len(a), maxalen)
        for qw in q:
            if qw not in qwords:
                qwords[qw] = 0
            qwords[qw] += 1
        for aw in a:
            if aw not in awords:
                awords[aw] = 0
            awords[aw] += 1
    qmat = np.zeros((len(qss), maxqlen), dtype="int32")
    amat = np.zeros((len(ass), maxalen), dtype="int32")
    qdic = dict(zip([x for x, y in sorted(qwords.items(), reverse=True, key=lambda (x, y): y)],
                    range(2, len(qwords) + 2)))
    adic = dict(zip([x for x, y in sorted(awords.items(), reverse=True, key=lambda (x, y): y)],
                    range(1, len(awords) + 1)))
    for i in range(len(qss)):
        q = qss[i]
        a = ass[i]
        qx = [qdic[x] for x in q]
        if reverse:
            qx.reverse()
        qmat[i, :len(q)] = qx
        amat[i, :len(a)] = [adic[x] for x in a]
    return qmat, amat, qdic, adic, qwords, awords


def loadgeo(trainp="../../../data/semparse/geoquery.lbd.dev",
            testp="../../../data/semparse/geoquery.lbd.test",
            customemb=False,
            reverse=True,
            transformer=None):

    d = []

    addlines(trainp, d, transformer=transformer)
    addlines(testp, d, transformer=transformer)

    return loadgeopl(p=d, customemb=customemb, reverse=reverse)


def fixbrackets(m):
    ret = ""
    if len(m.group(1)) > 0:
        ret += m.group(1)
        ret += " "
    ret += m.group(2)
    if len(m.group(3)) > 0:
        ret += " "
        ret += m.group(3)
    return ret


def addlines(p, d, transformer=None):
    curline = ""
    for line in open(p):
        if len(curline) == 0:
            curline = line
        else:
            if line == "\n":
                d.append("" + curline)
                curline = ""
            elif line[:2] == "//":
                pass
            else:
                oldline = line
                line = line[:-1]
                while oldline != line:
                    oldline = line
                    line = re.sub("([^\s]?)([()])([^\s]?)",
                                  fixbrackets,
                                  line)
                if transformer is not None:
                    line = transformer(line)
                curline = "{}\t{}".format(curline, line)


def loadgeoauto(p="../../../data/semparse/geoquery.lbd.autogen",
                reverse=True, transformer=None):
    d = []
    addlines(p, d, transformer=transformer)
    return loadgeopl(p=d, customemb=False, reverse=reverse, splitre="\s")


class VectorPosEmb(Block):
    def __init__(self, baseemb, numpos, posembdim, **kw):
        super(VectorPosEmb, self).__init__(**kw)
        self.baseemb = baseemb
        self.pemb = VectorEmbed(indim=numpos, dim=posembdim)
        self.outdim = self.baseemb.outdim + self.pemb.outdim
        self.indim = self.baseemb.indim

    def apply(self, x):     # (batsize, seqlen, 2)
        wembeddings = self.baseemb(x[:, :, 0])
        pembeddings = self.pemb(x[:, :, 1])
        ret = T.concatenate([wembeddings, pembeddings], axis=2)     # (batsize, seqlen, wembdim+pembdim)
        ret.mask = wembeddings.mask
        return ret


class SoftMaxOut(Block):
    def __init__(self, indim=None, innerdim=None, outvocsize=None, dropout=None, **kw):
        super(SoftMaxOut, self).__init__(**kw)
        self.indim, self.innerdim, self.outvocsize = indim, innerdim, outvocsize
        self.lin1 = Linear(indim=indim, dim=innerdim, dropout=dropout)
        self.lin2 = MatDot(indim=innerdim, dim=outvocsize)

    def apply(self, x):
        a = self.lin1(x)
        b = Tanh()(a)
        c = self.lin2(b)
        d = Softmax()(c)
        return d

    def setlin2(self, v):
        self.lin2 = MatDot(indim=self.indim, dim=self.innerdim, value=v)


def preprocess(qmat, amat, qdic, adic, qwc, awc, maskid, qreversed=False, dorare=True):
    # TODO: add positional replacement and change other functions accordingly
    amat[amat == adic["capital:c"]] = adic["capital:t"]
    replaceina = set()
    for k in adic:
        if (k[-2:] in ":c :s :r :m :n".split() or
            k[-3:] in ":lo :co".split()) and not k == "capital:c":
            replaceina.add(k)
    for r in replaceina:
        splits = r.split(":")
        rt = splits[1]+"-type"
        if not rt in adic:
            adic[rt] = max(adic.values()) + 1
        if not rt in qdic:
            qdic[rt] = max(qdic.values()) + 1
    radic = {v: k for k, v in adic.items()}
    rqdic = {v: k for k, v in qdic.items()}
    for i in range(qmat.shape[0]):
        if i == 379:
            pass
        for j in range(amat.shape[1]):
            if amat[i, j] in {adic[x] for x in replaceina}:
                sf = radic[amat[i, j]].split(":")[0].split("_")
                #if sf[-1] == "river" or len(sfs[0][-1]) == 2:
                #    sf = sf[:-1]
                sft = radic[amat[i, j]].split(":")[1]
                amat[i, j] = adic[sft+"-type"]
                sfs = [sf]
                qmati = qmat[i]
                if qreversed:
                    qmatio = maskid * np.ones_like(qmati)
                    m = qmati.shape[0] - 1
                    n = 0
                    while m >= 0:
                        if qmati[m] == maskid:
                            pass
                        else:
                            qmatio[n] = qmati[m]
                            n += 1
                        m -= 1
                    qmati = qmatio
                if sf == ["usa"]:
                    sfs.append("united states".split())
                    sfs.append("the country".split())
                    sfs.append("the states".split())
                    sfs.append(["us"])
                    sfs.append(["america"])
                for sf in sfs:
                    k = 0
                    done = False
                    while k < qmat.shape[1]:
                        if qmati[k] != maskid and \
                                        rqdic[qmati[k]] == sf[0]:
                            l = 0
                            while l < len(sf) and l + k < qmat.shape[1]:
                                if rqdic[qmati[k + l]] == sf[l]:
                                    l += 1
                                else:
                                    break
                            if l >= len(sf) - (1 if sf[0] != "the" else 0):
                                qmati[k] = qdic[sft+"-type"]
                                qmati[k+1:qmat.shape[1]-l+1] = qmati[k+l:]
                                qmati[qmat.shape[1]-l+1:] = maskid
                                done = True
                                break
                        k += 1
                    if done:
                        break
                if qreversed:
                    qmatio = maskid * np.ones_like(qmati)
                    m = qmati.shape[0] - 1
                    n = 0
                    while m >= 0:
                        if qmati[m] == maskid:
                            pass
                        else:
                            qmatio[n] = qmati[m]
                            n += 1
                        m -= 1
                    qmati = qmatio
                qmat[i] = qmati
    # test
    wop = []
    for i in range(qmat.shape[0]):
        if "-type" in wordids2string(amat[i], {v: k for k, v in adic.items()}) and \
            "-type" not in wordids2string(qmat[i], {v: k for k, v in qdic.items()}):
            wop.append(i)
    print "{}/{}".format(len(wop), qmat.shape[0])
    # rare words
    if dorare:
        rareset = set(map(lambda (x, y): x,
                          filter(lambda (x, y): y < 2,
                                 sorted(qwc.items(), key=lambda (x, y): y))))
        rareids = {qdic[x] for x in rareset}
        qmat = np.vectorize(lambda x: qdic["<RARE>"] if x in rareids else x)(qmat)
        def pp(i):
            print wordids2string(qmat[i], {v: k for k, v in qdic.items()})
            print wordids2string(amat[i], {v: k for k, v in adic.items()})
    #embed()

    return qmat, amat, qdic, adic, qwc, awc


def gentypdic(qdic, adic):
    rqdic = {v: k for k, v in qdic.items()}
    radic = {v: k for k, v in adic.items()}
    types = [k[:-5] for k in qdic if k[-5:] == "-type"]
    ret = {qdic[t+"-type"]: (adic[t+"-type"], {}) for t in types} # dict from qdic typid to ( adic type id, {adic entity id: [sfs for entity in qdic]}))
    for t in types:
        fls = [v for v in adic if (v[-len(t)-1:] == ":"+t and v != "capital:c")]
        typdicpart = {fl: [re.sub("_[a-z]{2}$", "", re.sub(":[a-z]{1,2}$", "", fl)).split("_")]
                      for fl in fls}
        if "usa:co" in typdicpart:
            typdicpart["usa:co"].append("united states".split())
            typdicpart["usa:co"].append("the states".split())
            typdicpart["usa:co"].append(["us"])
            typdicpart["usa:co"].append(["america"])
        idtypdicpart = {adic[k]: [[qdic[vew] for vew in ve] for ve in v] for k, v in typdicpart.items()}
        ret[qdic[t+"-type"]][1].update(idtypdicpart)
    return ret


def generate(qmat, amat, qdic, adic, oqmat, oamat, reversed=True):
    # !!! respect train test split
    import re
    qmat = np.insert(qmat, 0, np.max(qmat) * np.ones((qmat.shape[1])), axis=0)
    amat = np.insert(amat, 0, np.max(amat) * np.ones((amat.shape[1])), axis=0)
    tqstrings = [re.sub("(\s0)+\s?$", "", a) for a in list(np.apply_along_axis(lambda x: " ".join(map(str, x)), 1, qmat))][1:] #qmat[:-279]))][1:]
    tastrings = [re.sub("(\s0)+\s?$", "", a) for a in list(np.apply_along_axis(lambda x: " ".join(map(str, x)), 1, amat))][1:] #amat[:-279]))][1:]
    qmat = qmat[1:]
    amat = amat[1:]
    #oqmat = oqmat[-279:]
    #oamat = oamat[-279:]
    oqmat = oqmat[:279]
    oamat = oamat[:279]
    oqmat = np.insert(oqmat, 0, np.max(oqmat) * np.ones((oqmat.shape[1])), axis=0)
    oamat = np.insert(oamat, 0, np.max(oamat) * np.ones((oamat.shape[1])), axis=0)
    xqstrings = [re.sub("(\s0)+\s?$", "", a) for a in list(np.apply_along_axis(lambda x: " ".join(map(str, x)), 1, oqmat))][1:]
    xastrings = [re.sub("(\s0)+\s?$", "", a) for a in list(np.apply_along_axis(lambda x: " ".join(map(str, x)), 1, oamat))][1:]
    oqmat = oqmat[1:]
    oamat = oamat[1:]
    #embed()
    # generate dic from type ids to pairs of fl ids and seqs of word-ids
    gendic = gentypdic(qdic, adic)
    # make new training data, without overlap
    rqdic = {v: k for k, v in qdic.items()}
    radic = {v: k for k, v in adic.items()}
    def pp(x):
        print wordids2string([int(xe) for xe in x[0].split()], rqdic, maskid=0)
        print wordids2string([int(xe) for xe in x[1].split()], radic, maskid=0)
    newt = []
    for i in range(len(tqstrings)):
        qss = tqstrings[i]
        ass = tastrings[i]
        if i == 379:
            pass #embed()
        acc = []
        saveacc = []
        newacc = [(qss, ass)]
        while len(newacc) > 0:    # fill next slot of each on in acc
            acc = newacc
            newacc = []
            while len(acc) > 0:
                qss, ass = acc.pop()
                k = 0
                qssids = map(int, qss.split())
                while k < len(qssids) and qssids[k] not in gendic.keys(): # find next slot and fill
                    k += 1
                if k < len(qssids):
                    id = qssids[k]
                else:
                    saveacc.append((qss, ass))
                    continue
                l = 0
                assids = map(int, ass.split())
                try:
                    while assids[l] != gendic[id][0]:
                        l += 1
                except IndexError, e:
                    pass
                    embed()
                for fl, sfs in gendic[id][1].items():
                    assids = map(int, ass.split())
                    assids.pop(l)
                    assids.insert(l, fl)
                    wids = [[sfw for sfw in sf] for sf in sfs]
                    for widse in wids:
                        if reversed:
                            widse.reverse()
                        m = 0
                        qssids = map(int, qss.split())
                        qssids.pop(k)
                        for widses in widse:
                            qssids.insert(k + m, widses)
                            m += 1
                        newacc.append((" ".join(map(str, qssids)), " ".join(map(str, assids))))
        newt.extend(saveacc)
    newt = list(set(newt))
    allx = set(zip(xqstrings, xastrings))
    newt = list(set(newt).difference(allx))
    newt.extend(allx)
    maxqlen, maxalen = reduce(lambda (a, b), (c, d): (max(a, c), max(b, d)), [(len(x.split()), len(y.split())) for (x, y) in newt], (0, 0))
    #maxalen = reduce(lambda x, y: max(x, y), [len(y) for (x, y) in newt], 0)
    #embed()
    # make matrices
    newtqmat = np.zeros((len(newt), maxqlen), dtype="int32")
    newtamat = np.zeros((len(newt), maxalen), dtype="int32")
    for i in range(len(newt)):
        qids = [int(x) for x in newt[i][0].split()]
        aids = [int(x) for x in newt[i][1].split()]
        newtqmat[i, :len(qids)] = qids
        newtamat[i, :len(aids)] = aids

    def ppp(i=None):
        def pp(i):
            print wordids2string(newtqmat[i], rqdic, 0)
            print wordids2string(newtamat[i], radic, 0)
        if i is None:
            for i in range(newtqmat.shape[0]):
                pp(i)
                raw_input()
        else:
            pp(i)
    print "{} examples after generation".format(newtqmat.shape[0])
    #embed()
    return newtqmat, newtamat


class GenSample(object):
    def __init__(self, td, reversed=True):
        self.typdic = td
        self.reversed = reversed

    def __call__(self, encinp, decinp, gold, phase=None):
        if phase == "TRAIN" and self.typdic is not None:  # replace a type id with entity id in all inputs
            encacc = []
            decacc = []
            goldacc = []
            for i in range(encinp.shape[0]):
                encinprow = list(encinp[i])
                decinprow = list(decinp[i])
                goldrow = list(gold[i])
                holdersleft = True
                while holdersleft:
                    j = 0
                    while encinprow[j] not in self.typdic:
                        j += 1
                        if j == len(encinprow):  # at the end
                            holdersleft = False
                            break
                    if not holdersleft:
                        continue
                    k = 1
                    while decinprow[k] != self.typdic[encinprow[j]][0]:
                        k += 1
                        if k == len(decinprow):
                            raise Exception("not found")
                    fl = random.choice(self.typdic[encinprow[j]][1].keys())  # choose random filling
                    wids = random.choice(self.typdic[encinprow[j]][1][fl])  # choose random verbalization of filling
                    if self.reversed:
                        wids.reverse()
                    decinprow[k] = fl
                    goldrow[k - 1] = fl
                    encinphead = list(encinprow[:j])
                    encinptail = list(encinprow[j + 1:])
                    encinprow = encinphead + wids + encinptail
                encacc.append(encinprow)
                decacc.append(decinprow)
                goldacc.append(goldrow)
            encmaxlen = reduce(max, map(len, encacc), 0)
            decmaxlen = reduce(max, map(len, decacc), 0)
            goldmaxlen = reduce(max, map(len, goldacc), 0)
            encout = np.zeros((encinp.shape[0], encmaxlen), dtype="int32")
            decout = np.zeros((decinp.shape[0], decmaxlen), dtype="int32")
            goldout = np.zeros((gold.shape[0], goldmaxlen), dtype="int32")
            for i, (e, d, g) in enumerate(zip(encacc, decacc, goldacc)):
                encout[i, :len(e)] = e
                decout[i, :len(d)] = d
                goldout[i, :len(g)] = g
            # embed()
            return encout, decout, goldout
        else:
            return encinp, decinp, gold


def add_pos_indexes(qmat, oqmat, amat, oamat):
    qposmat = np.arange(0, qmat.shape[1])[None, :]
    qposmat = np.repeat(qposmat, qmat.shape[0], axis=0)
    qmat = np.concatenate([qmat[:, :, None], qposmat[:, :, None]], axis=2)
    oqmat = np.concatenate([oqmat[:, :, None], qposmat[:, :, None]], axis=2)
    aposmat = np.arange(0, amat.shape[1])[None, :]
    aposmat = np.repeat(aposmat, amat.shape[0], axis=0)
    amati = np.concatenate([amat[:, :, None], aposmat[:, :, None]], axis=2)
    oamati = np.concatenate([oamat[:, :, None], aposmat[:, :, None]], axis=2)
    return qmat, oqmat, amat, oamat


class RandomCorrupt(object):
    def __init__(self, p=0.1, corruptdecoder=None, corruptencoder=None, maskid=0):
        self.corruptdecoder = corruptdecoder  # decoder corrupt range
        self.corruptencoder = corruptencoder  # encoder corrupt range
        self.p = p
        self.maskid = maskid

    def __call__(self, encinp, decinp, gold, phase=None):
        if phase == "TRAIN":
            if self.corruptencoder is not None:
                encinp = self._corruptseq(encinp, self.corruptencoder)
            if self.corruptdecoder is not None:
                decinp = self._corruptseq(decinp, self.corruptdecoder)
        return encinp, decinp, gold

    def _corruptseq(self, seq, range):
        if self.p > 0:
            corrupt = np.random.randint(range[0], range[1], seq.shape, dtype="int32")
            mask = np.random.random(seq.shape) < self.p
            seqmask = seq != self.maskid
            mask = np.logical_and(mask, seqmask)
            outp = ((1 - mask) * seq + mask * corrupt).astype("int32")
            # embed()
        else:
            outp = seq
        return outp


def to_char_level(qmat, amat, qdic, adic, maskid):
    qmat = wordmat2charmat(qmat, qdic, maxlen=1000, maskid=maskid)
    amat = wordmat2charmat(amat, adic, maxlen=1000, maskid=maskid)
    qmat[qmat > 0] += 2
    amat[amat > 0] += 2
    qdic = dict([(chr(x), x + 2) for x in range(np.max(qmat))])
    adic = dict([(chr(x), x + 2) for x in range(np.max(amat))])
    qdic.update({"<RARE>": 1})
    adic.update({"<RARE>": 1})
    print wordids2string(qmat[0], {v: k for k, v in qdic.items()})
    print wordids2string(amat[0], {v: k for k, v in adic.items()})
    return qmat, amat, qdic, adic


def compute_overlap(qmat_t, amat_t, qmat_x, amat_x):
    def to_mat_strings(mat):
        matfs = np.insert(mat, [0], np.max(mat) * np.ones_like(mat[0]), axis=0)
        return np.apply_along_axis(lambda x: " ".join([str(xe) for xe in list(x)])[1:], 1, matfs)
    qmatstrings_t, qmatstrings_x, amatstrings_t, amatstrings_x = \
        map(to_mat_strings, [qmat_t, qmat_x, amat_t, amat_x])
    matstrings_t, matstrings_x = map(lambda a: map(lambda (x, y): x + y, a), [zip(qmatstrings_t, amatstrings_t), zip(qmatstrings_x, amatstrings_x)])
    qoverlap = set(qmatstrings_t).intersection(set(qmatstrings_x))
    aoverlap = set(amatstrings_t).intersection(set(amatstrings_x))
    overlap = set(matstrings_t).intersection(set(matstrings_x))
    return qoverlap, aoverlap, overlap


def do_custom_emb(inpemb, outemb, awc, embdim):
    thresh = 10
    sawc = sorted(awc.items(), key=lambda (k, v): v, reverse=True)
    rarewords = {k for (k, v) in sawc if v < thresh}
    g = Glove(embdim)
    inpemb = inpemb.override(g)
    outemb = outemb.override(g, which=rarewords)
    return inpemb, outemb


def split_train_test(mat, sep=-279):
    return mat[:sep], mat[sep:]


def run(
        numbats=50,
        epochs=10,
        lr=0.5,
        embdim=50,
        encdim=400,
        dropout=0.2,
        layers=1,
        inconcat=True,
        outconcat=True,
        posemb=False,
        customemb=False,
        preproc="none",     # "none" or "generate" or "abstract" or "gensample"
        bidir=False,
        corruptnoise=0.0,
        inspectdata=False,
        relinearize="none",
        pretrain=False,
        pretrainepochs=-1,
        pretrainnumbats=-1,
        pretrainlr=-0.1,
        loadpretrained="none",
        wreg=0.0,
        testmode=False):

    #TODO: bi-encoder and other beasts
    #TODO: make sure gensample results NOT IN test data

    if pretrain == True:
        assert(preproc == "none" or preproc == "gensample")
        pretrainepochs = epochs if pretrainepochs == -1 else pretrainepochs

    ######### DATA LOADING AND TRANSFORMATIONS ###########
    srctransformer = None
    if relinearize != "none":
        lambdaparser = LambdaParser()
        if relinearize == "greedy":
            def srctransformer(x): return lambdaparser.parse(x).greedy_linearize(deeppref=True)
        elif relinearize == "deep":
            def srctransformer(x): return lambdaparser.parse(x).deep_linearize()
        else:
            raise Exception("unknown linearization")

    if pretrain:        ### PRETRAIN DATA LOAD ###
        qmat_auto, amat_auto, qdic_auto, adic_auto, qwc_auto, awc_auto = \
            loadgeoauto(reverse=True, transformer=srctransformer)
        def pp(i):
            print wordids2string(qmat_auto[i], {v: k for k, v in qdic_auto.items()}, 0)
            print wordids2string(amat_auto[i], {v: k for k, v in adic_auto.items()}, 0)
        if inspectdata:
            print "pretrain inspect"
            embed()
    qmat, amat, qdic, adic, qwc, awc = loadgeo(customemb=customemb, reverse=True, transformer=srctransformer)

    maskid = 0
    typdic = None
    oqmat = qmat.copy()
    oamat = amat.copy()
    print "{} is preproc".format(preproc)
    if preproc != "none":
        qmat, amat, qdic, adic, qwc, awc = preprocess(qmat, amat, qdic, adic, qwc, awc, maskid, qreversed=True, dorare=preproc != "generate")
        if preproc == "generate":   # alters size
            print "generating"
            qmat, amat = generate(qmat, amat, qdic, adic, oqmat, oamat, reversed=True)
            #embed()
        elif preproc == "gensample":
            typdic = gentypdic(qdic, adic)

    ######### train/test split from here #########
    qmat_t, qmat_x = split_train_test(qmat)
    amat_t, amat_x = split_train_test(amat)
    oqmat_t, oqmat_x = split_train_test(oqmat)
    oamat_t, oamat_x = split_train_test(oamat)

    qoverlap, aoverlap, overlap = compute_overlap(qmat_t, amat_t, qmat_x, amat_x)
    print "overlaps: {}, {}: {}".format(len(qoverlap), len(aoverlap), len(overlap))

    if inspectdata:
        embed()

    np.random.seed(12345)

    encdimi = [encdim/2 if bidir else encdim] * layers
    decdimi = [encdim] * layers

    amati_t, amati_x = amat_t, amat_x
    oamati_t, oamati_x = oamat_t, oamat_x
    if pretrain:
        amati_auto = amat_auto

    if posemb:      # add positional indexes to datamatrices
        qmat_t, oqmat_t, amat_t, oamat_t = add_pos_indexes(qmat_t, oqmat_t, amat_t, oamat_t)
        qmat_x, oqmat_x, amat_x, oamat_x = add_pos_indexes(qmat_x, oqmat_x, amat_x, oamat_x)

    if preproc == "gensample":
        qmat_x, amat_x, amati_x = oqmat_x, oamat_x, oamati_x

    rqdic = {v: k for k, v in qdic.items()}
    radic = {v: k for k, v in adic.items()}

    def tpp(i):
        print wordids2string(qmat_t[i], rqdic, 0)
        print wordids2string(amat_t[i], radic, 0)

    def xpp(i):
        print wordids2string(qmat_x[i], rqdic, 0)
        print wordids2string(amat_x[i], radic, 0)

    if inspectdata:
        embed()
    print "{} training examples".format(qmat_t.shape[0])

    ################## MODEL DEFINITION ##################
    # encdec prerequisites
    inpemb = WordEmb(worddic=qdic, maskid=maskid, dim=embdim)
    outemb = WordEmb(worddic=adic, maskid=maskid, dim=embdim)

    if pretrain == True:
        inpemb_auto = WordEmb(worddic=qdic_auto, maskid=maskid, dim=embdim)
        outemb = WordEmb(worddic=adic_auto, maskid=maskid, dim=embdim)

    if customemb:
        inpemb, outemb = do_custom_emb(inpemb, outemb, awc, embdim)
        if pretrain:
            inpemb_auto, outemb = do_custom_emb(inpemb_auto, outemb, awc_auto, embdim)

    if posemb:  # use custom emb layers, with positional embeddings
        posembdim = 50
        inpemb = VectorPosEmb(inpemb, qmat_t.shape[1], posembdim)
        outemb = VectorPosEmb(outemb, amat_t.shape[1], posembdim)
        if pretrain:
            inpemb_auto = VectorPosEmb(inpemb_auto, qmat_auto.shape[1], posembdim)
            outemb = VectorPosEmb(outemb, max(amat_auto.shape[1], amat_t.shape[1]), posembdim)

    smodim = embdim
    smo = SoftMaxOut(indim=encdim + encdim, innerdim=smodim,
                     outvocsize=len(adic) + 1, dropout=dropout)

    if customemb:
        smo.setlin2(outemb.baseemb.W.T)

    # encdec model
    encdec = SimpleSeqEncDecAtt(inpvocsize=max(qdic.values()) + 1,
                                inpembdim=embdim,
                                inpemb=inpemb,
                                outvocsize=max(adic.values()) + 1,
                                outembdim=embdim,
                                outemb=outemb,
                                encdim=encdimi,
                                decdim=decdimi,
                                maskid=maskid,
                                statetrans=True,
                                dropout=dropout,
                                inconcat=inconcat,
                                outconcat=outconcat,
                                rnu=GRU,
                                vecout=smo,
                                bidir=bidir,
                                )

    ################## TRAINING ##################
    if pretrain == True or loadpretrained != "none":
        if pretrain == True and loadpretrained == "none":
            '''encdec.remake_encoder(inpvocsize=max(qdic_auto.values()) + 1,
                                  inpembdim=embdim,
                                  inpemb=inpemb_auto,
                                  maskid=maskid,
                                  dropout_h=dropout,
                                  dropout_in=dropout)
                                  '''
            encdec.enc.embedder = inpemb_auto
        if loadpretrained != "none":
            encdec = encdec.load(loadpretrained+".pre.sp.model")
            print "MODEL LOADED: {}".format(loadpretrained)
        if pretrain == True:
            if pretrainnumbats < 0:
                import math
                batsize = int(math.ceil(qmat_t.shape[0] * 1.0 / numbats))
                pretrainnumbats = int(math.ceil(qmat_auto.shape[0] * 1.0 / batsize))
                print "{} batches".format(pretrainnumbats)
            if pretrainlr < 0:
                pretrainlr = lr
            if testmode:
                oldparamvals = {p: p.v for p in encdec.get_params()}
                qmat_auto = qmat_auto[:100]
                amat_auto = amat_auto[:100]
                amati_auto = amati_auto[:100]
                pretrainnumbats = 10
            #embed()
            encdec.train([qmat_auto, amat_auto[:, :-1]], amati_auto[:, 1:])\
                .cross_entropy().adadelta(lr=pretrainlr).grad_total_norm(5.) \
                .l2(wreg).exp_mov_avg(0.95) \
                .split_validate(splits=10, random=True).cross_entropy().seq_accuracy() \
                .train(pretrainnumbats, pretrainepochs)

            if testmode:
                for p in encdec.get_params():
                    print np.linalg.norm(p.v - oldparamvals[p], ord=1)
            savepath = "{}.pre.sp.model".format(random.randint(1000, 9999))
            print "PRETRAIN SAVEPATH: {}".format(savepath)
            encdec.save(savepath)


        # NaN somewhere at 75% in training, in one of RNU's? --> with rmsprop

        '''encdec.remake_encoder(inpvocsize=max(qdic.values()) + 1,
                              inpembdim=embdim,
                              inpemb=inpemb,
                              maskid=maskid,
                              dropout_h=dropout,
                              dropout_in=dropout)
        encdec.dec.set_lr(0.0)
        '''
        encdec.enc.embedder = inpemb

    encdec.train([qmat_t, amat_t[:, :-1]], amati_t[:, 1:])\
        .sampletransform(GenSample(typdic),
                         RandomCorrupt(corruptdecoder=(2, max(adic.values()) + 1),
                                       corruptencoder=(2, max(qdic.values()) + 1),
                                       maskid=maskid, p=corruptnoise))\
        .cross_entropy().adadelta(lr=lr).grad_total_norm(5.) \
        .l2(wreg).exp_mov_avg(0.8) \
        .validate_on([qmat_x, amati_x[:, :-1]], amat_x[:, 1:]) \
        .cross_entropy().seq_accuracy()\
        .train(numbats, epochs)
    #.split_validate(splits=10, random=True)\

    qrwd = {v: k for k, v in qdic.items()}
    arwd = {v: k for k, v in adic.items()}

    def play(*x, **kw):
        hidecorrect = False
        if "hidecorrect" in kw:
            hidecorrect = kw["hidecorrect"]
        if len(x) == 1:
            x = x[0]
            q = wordids2string(qmat_x[x], rwd=qrwd, maskid=maskid, reverse=True)
            ga = wordids2string(amat_x[x, 1:], rwd=arwd, maskid=maskid)
            pred = encdec.predict(qmat_x[x:x+1], amati_x[x:x+1, :-1])
            pa = wordids2string(np.argmax(pred[0], axis=1), rwd=arwd, maskid=maskid)
            if hidecorrect and ga == pa[:len(ga)]:  # correct
                return False
            else:
                print "{}: {}".format(x, q)
                print ga
                print pa
                return True
        elif len(x) == 0:
            for i in range(0, qmat_x.shape[0]):
                r = play(i)
                if r:
                    raw_input()
        else:
            raise Exception("invalid argument to play")
    embed()


if __name__ == "__main__":
    argprun(run)