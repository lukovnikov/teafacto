from teafacto.util import argprun, isstring, issequence
from teafacto.procutil import wordids2string, wordmat2charmat
import numpy as np, re
from IPython import embed

from teafacto.core.base import Val, Block, tensorops as T
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt
from teafacto.blocks.basic import VectorEmbed, Linear, MatDot
from teafacto.blocks.activations import Softmax, Tanh
from teafacto.blocks.lang.wordvec import WordEmb, Glove


def loadgeopl(p="../../../data/semparse/geoquery.txt", customemb=False, reverse=True):
    qss, ass = [], []
    maxqlen, maxalen = 0, 0
    qwords, awords = {"<RARE>": 1}, {}

    if isstring(p):
        p = open(p)

    for line in p:
        splitre = "[\s-]" if customemb else "\s"
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
            reverse=True):

    d = []

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

    def addlines(p, d):
        curline = ""
        for line in open(p):
            if len(curline) == 0:
                curline = line
            else:
                if line == "\n":
                    d.append(""+curline)
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
                    curline = "{}\t{}".format(curline, line)

    addlines(trainp, d)
    addlines(testp, d)

    return loadgeopl(p=d, customemb=customemb, reverse=reverse)


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
                        k += 1
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


def generate(qmat, amat, qdic, adic, oqmat, oamat, reversed=True):
    # !!! respect train test split
    import re
    qmat = np.insert(qmat, 0, np.max(qmat) * np.ones((qmat.shape[1])), axis=0)
    amat = np.insert(amat, 0, np.max(amat) * np.ones((amat.shape[1])), axis=0)
    tqstrings = [re.sub("(\s0)+\s?$", "", a) for a in list(np.apply_along_axis(lambda x: " ".join(map(str, x)), 1, qmat[:-279]))][1:]
    tastrings = [re.sub("(\s0)+\s?$", "", a) for a in list(np.apply_along_axis(lambda x: " ".join(map(str, x)), 1, amat[:-279]))][1:]
    qmat = qmat[1:]
    amat = amat[1:]
    oqmat = oqmat[-279:]
    oamat = oamat[-279:]
    oqmat = np.insert(oqmat, 0, np.max(oqmat) * np.ones((oqmat.shape[1])), axis=0)
    oamat = np.insert(oamat, 0, np.max(oamat) * np.ones((oamat.shape[1])), axis=0)
    xqstrings = [re.sub("(\s0)+\s?$", "", a) for a in list(np.apply_along_axis(lambda x: " ".join(map(str, x)), 1, oqmat))][1:]
    xastrings = [re.sub("(\s0)+\s?$", "", a) for a in list(np.apply_along_axis(lambda x: " ".join(map(str, x)), 1, oamat))][1:]
    oqmat = oqmat[1:]
    oamat = oamat[1:]
    #embed()
    # generate dic from type ids to pairs of fl ids and seqs of word-ids
    types = [k for k in qdic if k[-5:] == "-type"]
    flspertype = {k: [v for v in adic
                        if (v[-len(k[:-5])-1:] == ":"+k[:-5]
                            and v != "capital:c")
                      ]
                  for k in types}
    # make new training data, without overlap
    rqdic = {v: k for k, v in qdic.items()}
    radic = {v: k for k, v in adic.items()}
    newt = []
    for i in range(len(tqstrings)):
        qss = tqstrings[i]
        ass = tastrings[i]
        typeid = set(qss.split()).intersection({str(qdic[ts]) for ts in types})
        if len(typeid) == 1:
            typeid = list(typeid)[0]
            for fl in flspertype[rqdic[int(typeid)]]:
                flid = str(adic[fl])
                words = [re.sub("_[a-z]{2}$", "", re.sub(":[a-z]{1,2}$", "", fl)).split("_")]
                if words[0] == ["usa"]:
                    words.append("united states".split())
                    words.append(["us"])
                    words.append(["america"])
                    words.append("the states".split())
                for wordse in words:
                    wordse.reverse()
                    nlids = " ".join([str(qdic[x]) for x in wordse])
                    nqss = qss.replace(" " + typeid + " ", " " + nlids + " ")
                    nass = ass.replace(" " + str(adic[rqdic[int(typeid)]]) + " ", " " + flid + " ")
                    newt.append((nqss, nass))
    newt = list(set(newt))
    def pp(x):
        print wordids2string([int(xe) for xe in x[0].split()], rqdic, maskid=0)
        print wordids2string([int(xe) for xe in x[1].split()], radic, maskid=0)
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
    embed()
    return newtqmat, newtamat


def run(
        numbats=50,
        epochs=10,
        lr=1.,
        embdim=50,
        encdim=400,
        dropout=0.2,
        layers=1,
        inconcat=True,
        outconcat=True,
        posemb=False,
        customemb=False,
        charlevel=False,
        preproc="abstract",     # or "none" or "generate" or "abstract"
        bidir=False,
        corruptnoise=0.0):

    #TODO: with generation, max length is smaller than without generation
    # ===> TODO: SUM TING WONG!!!
    #TODO: bi-encoder and other beasts
    # loaddata
    qmat, amat, qdic, adic, qwc, awc = loadgeo(customemb=customemb, reverse=not charlevel)
    qmatstrings = np.apply_along_axis(lambda x: " ".join([str(xe) for xe in list(x)]), 1, qmat)
    amatstrings = np.apply_along_axis(lambda x: " ".join([str(xe) for xe in list(x)]), 1, amat)
    matstrings = [x + y for x, y in zip(qmatstrings, amatstrings)]
    qoverlap = set(qmatstrings[:600]).intersection(set(qmatstrings[600:]))
    aoverlap = set(amatstrings[:600]).intersection(set(amatstrings[600:]))
    overlap = set(matstrings[:600]).intersection(set(matstrings[600:]))
    print "overlaps: {}, {}: {} / {}".format(len(qoverlap), len(aoverlap), len(overlap), len(amatstrings[600:]))
    #embed()
    maskid = 0
    print "{} is preproc".format(preproc)
    if preproc != "none":
        oqmat = qmat.copy()
        oamat = amat.copy()
        qmat, amat, qdic, adic, qwc, awc = preprocess(qmat, amat, qdic, adic, qwc, awc, maskid, qreversed=not charlevel, dorare=preproc != "generate")
        if preproc == "generate":
            print "generating"
            qmat, amat = generate(qmat, amat, qdic, adic, oqmat, oamat, reversed=not charlevel)
            rqdic = {v: k for k, v in qdic.items()}
            radic = {v: k for k, v in adic.items()}
            def pp(i):
                print wordids2string(qmat[i], rqdic, 0)
                print wordids2string(amat[i], radic, 0)
            #embed()

    if charlevel:
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

    np.random.seed(12345)

    #embed()

    encdimi = [encdim/2 if bidir else encdim] * layers
    decdimi = [encdim] * layers


    inpemb = WordEmb(worddic=qdic, maskid=maskid, dim=embdim)
    outemb = WordEmb(worddic=adic, maskid=maskid, dim=embdim)

    if customemb:
        thresh = 10
        sawc = sorted(awc.items(), key=lambda (k, v): v, reverse=True)
        rarewords = {k for (k, v) in sawc if v < thresh}
        g = Glove(embdim)
        inpemb = inpemb.override(g)
        outemb = outemb.override(g, which=rarewords)

    if posemb:      # custom emb layers, with positional embeddings
        posembdim = 50
        inpemb = VectorPosEmb(inpemb, qmat.shape[1], posembdim)
        outemb = VectorPosEmb(outemb, amat.shape[1], posembdim)

    smodim = embdim
    smo = SoftMaxOut(indim=encdim+encdim, innerdim=smodim,
                     outvocsize=len(adic)+1, dropout=dropout)

    if customemb:
        smo.setlin2(outemb.baseemb.W.T)

    # make seq/dec+att
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

    amati = amat

    class RandomCorrupt(object):
        def __init__(self, p = 0.1, corruptdecoder=None, corruptencoder=None, maskid=0):
            self.corruptdecoder = corruptdecoder    # decoder corrupt range
            self.corruptencoder = corruptencoder    # encoder corrupt range
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
                #embed()
            else:
                outp = seq
            return outp




    if posemb:
        qposmat = np.arange(0, qmat.shape[1])[None, :]
        qposmat = np.repeat(qposmat, qmat.shape[0], axis=0)
        qmat = np.concatenate([qmat[:, :, None], qposmat[:, :, None]], axis=2)
        aposmat = np.arange(0, amat.shape[1])[None, :]
        aposmat = np.repeat(aposmat, amat.shape[0], axis=0)
        amati = np.concatenate([amat[:, :, None], aposmat[:, :, None]], axis=2)

    tqmat = qmat[:-279]
    tamat = amat[:-279]
    tamati = amati[:-279]
    xqmat = qmat[-279:]
    xamat = amat[-279:]
    xamati = amati[-279:]

    #embed()
    print "{} training examples".format(tqmat.shape[0])

    encdec.train([tqmat, tamati[:, :-1]], tamat[:, 1:])\
        .sampletransform(RandomCorrupt(corruptdecoder=(2, max(adic.values()) + 1),
                                       corruptencoder=(2, max(qdic.values()) + 1),
                                       maskid=maskid, p=corruptnoise))\
        .cross_entropy().rmsprop(lr=lr/numbats).grad_total_norm(1.)\
        .split_validate(splits=5, random=True)\
        .cross_entropy().seq_accuracy()\
        .train(numbats, epochs)
    # .validate_on([xqmat, xamati[:, :-1]], xamat[:, 1:])\

    qrwd = {v: k for k, v in qdic.items()}
    arwd = {v: k for k, v in adic.items()}

    def play(*x, **kw):
        hidecorrect = False
        if "hidecorrect" in kw:
            hidecorrect = kw["hidecorrect"]
        if len(x) == 1:
            x = x[0]
            q = wordids2string(xqmat[x], rwd=qrwd, maskid=maskid, reverse=True)
            ga = wordids2string(xamat[x, 1:], rwd=arwd, maskid=maskid)
            pred = encdec.predict(xqmat[x:x+1], xamati[x:x+1, :-1])
            pa = wordids2string(np.argmax(pred[0], axis=1), rwd=arwd, maskid=maskid)
            if hidecorrect and ga == pa[:len(ga)]:  # correct
                return False
            else:
                print "{}: {}".format(x, q)
                print ga
                print pa
                return True
        elif len(x) == 0:
            for i in range(0, xqmat.shape[0]):
                r = play(i)
                if r:
                    raw_input()
        else:
            raise Exception("invalid argument to play")

    embed()

if __name__ == "__main__":
    argprun(run)