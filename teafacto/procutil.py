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
