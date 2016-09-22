import numpy as np


def buildwordmat(worddic, maxwordlen=30):
    maskid = -1
    rwd = sorted(worddic.items(), key=lambda (x, y): y)
    realmaxlen = 0
    wordmat = np.ones((rwd[-1][1]+1, maxwordlen), dtype="int32") * maskid
    for i in range(len(rwd)):
        rwdichars, rwdiidx = rwd[i]
        realmaxlen = max(realmaxlen, len(rwdichars))
        wordmat[rwdiidx, :min(len(rwdichars), maxwordlen)] \
            = [ord(c) for c in rwdichars[:min(len(rwdichars), maxwordlen)]]
    allchars = set(list(np.unique(wordmat))).difference({maskid})
    chardic = {maskid: maskid}
    chardic.update(dict(zip(allchars, range(len(allchars)))))
    wordmat = np.vectorize(lambda x: chardic[x])(wordmat)
    del chardic[maskid]
    chardic = {chr(k): v for k, v in chardic.items()}
    return wordmat, chardic


def wordmat2wordchartensor(wordmat, worddic=None, maxchars=30, maskid=-1):
    chartensor = wordmat2chartensor(wordmat, worddic=worddic, maxchars=maxchars, maskid=maskid)
    out = np.concatenate([wordmat[:, :, np.newaxis], chartensor], axis=2)
    #embed()
    return out


def wordmat2chartensor(wordmat, worddic=None, maxchars=30, maskid=-1):
    rwd = {v: k for k, v in worddic.items()}
    wordcharmat = maskid * np.ones((max(rwd.keys())+1, maxchars), dtype="int32")
    for i in rwd.keys():
        word = rwd[i]
        word = word[:min(maxchars, len(word))]
        wordcharmat[i, :len(word)] = [ord(ch) for ch in word]
    chartensor = wordcharmat[wordmat, :]
    chartensor[wordmat == -1] = -1
    return chartensor
