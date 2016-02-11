from teafacto.index.wikiindex import WikipediaIndex
from teafacto.qa.multichoice.kaggleutils import *
from teafacto.core.utils import ticktock, argparsify
import multiprocessing as mp, pandas as pd, math


def run(irp="../../../data/wikipedia/npageidx/",
        load=0,
        parallel=3,
        dataset="../../../data/kaggleai/training_set.tsv",
        prefix="t"):
    tmpsdfp = prefix+"sdf.tmp.csv"
    tmpcdfp = prefix+"preds.csv"
    df = read(path=dataset)
    if load == 0:
        tdf = transform(df)
        sdf = scoredf(tdf, irp, parallel=parallel)
        print sdf
        sdf.to_csv(tmpsdfp)
    else:
        sdf = pd.DataFrame.from_csv(tmpsdfp)
    cdf = choose(sdf)
    print cdf
    cdf.to_csv(tmpcdfp)
    res = evalu(cdf, df)
    print res


def transform(df):
    df["question"] = df["question"].apply(lambda x: " ".join(x))
    df["answerA"] = df["answerA"].apply(lambda x: " ".join(x))
    df["answerB"] = df["answerB"].apply(lambda x: " ".join(x))
    df["answerC"] = df["answerC"].apply(lambda x: " ".join(x))
    df["answerD"] = df["answerD"].apply(lambda x: " ".join(x))
    tdf = pd.DataFrame()
    tdf["qA"] = df.apply(lambda row: row["question"] + " " + row["answerA"], axis=1)
    tdf["qB"] = df.apply(lambda row: row["question"] + " " + row["answerB"], axis=1)
    tdf["qC"] = df.apply(lambda row: row["question"] + " " + row["answerC"], axis=1)
    tdf["qD"] = df.apply(lambda row: row["question"] + " " + row["answerD"], axis=1)
    tdf.index = df.index
    return tdf


def scoredf(tdf, irp, parallel=1):
    tt2 = ticktock("Scorer global")
    tt2.tick("scoring")
    if parallel > 1: # do parallellized
        chunks = parallel
        datalen = tdf.shape[0]
        chunksize = int(math.ceil(datalen*1.0/chunks))
        xdfcs = []
        for i in range(chunks):
            pxdf = tdf.iloc[i*chunksize: min((i+1)*chunksize, datalen)]
            xdfcs.append(pxdf)
        pool = mp.Pool()
        res = pool.map(scoredffun, zip(xdfcs, [{"irp": irp}]*len(xdfcs)))
        sdf = pd.concat(res)
        ####sdf = tdf.apply(lambda row: scorerow(row, scorer), axis=1)
    else:
        sdf = scoredffun((tdf, irp))
    sdf.index = tdf.index
    return sdf


def scoredffun((tdf, settings)):
    c = 0
    retdf = pd.DataFrame()
    tt = ticktock("Score fun")
    tt.tick("scorer initialized")
    scorer = elemscorer(WikipediaIndex(dir=settings["irp"]))
    for i, row in tdf.iterrows():
        c += 1
        if c % 10 == 0:
            tt.tock("%d/%d" % (c, tdf.shape[0])).tick()
            #break
        retdf = retdf.append(scorerow(row, scorer))
    tt.tock("scored")
    return retdf


def scorerow(row, scorer):
    return row.apply(lambda x: scorer(x))


def elemscorer(idx):
    def scorer(el):
        res = idx.search(el, limit=1)
        if len(res) > 0:
            res = res[0]
            return res["score"]
        else:
            return 0.0
    return scorer


def choose(sdf): # sdf only has qA, qB, qC, qD and the right index (question id's)
    ret = pd.DataFrame()
    ret["correctAnswer"] = sdf.idxmax(axis=1).apply(lambda x: x[1])
    ret.index = sdf.index
    return ret


def evalu(pred, orig):
    return (pred["correctAnswer"] == orig["correctAnswer"]).sum(axis=0)*1. / orig.shape[0]


# TODO: make incomplete retrieval (now many things are scored at 0.0)

if __name__ == "__main__":
    run(**argparsify(run))
