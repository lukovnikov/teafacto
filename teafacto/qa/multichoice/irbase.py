from teafacto.index.wikipedia import WikipediaIndex
from teafacto.qa.multichoice.kaggleutils import *
from teafacto.core.utils import ticktock


def run(irwiki, load=False):
    tmpsdfp = "sdf.tmp.csv"
    tmpcdfp = "preds.csv"
    df = read()
    if not load:
        tdf = transform(df)
        irscorer = elemscorer(irwiki)
        sdf = scoredf(tdf, irscorer)
        print sdf
        sdf.to_csv(tmpsdfp)
    else:
        sdf = pd.DataFrame.from_csv(tmpsdfp)
    cdf = choose(sdf)
    cdf.to_csv(tmpcdfp)
    evalu(cdf, df)
    print sdf


def transform(df):
    df["question"] = df["question"].apply(lambda x: " ".join(x))
    df["answerA"] = df["answerA"].apply(lambda x: " ".join(x))
    df["answerB"] = df["answerB"].apply(lambda x: " ".join(x))
    df["answerC"] = df["answerC"].apply(lambda x: " ".join(x))
    df["answerD"] = df["answerD"].apply(lambda x: " ".join(x))
    df["x"] = df.apply(lambda row: " ".join(row), axis=1)
    tdf = pd.DataFrame()
    tdf["qA"] = df.apply(lambda row: row["question"] + " " + row["answerA"], axis=1)
    tdf["qB"] = df.apply(lambda row: row["question"] + " " + row["answerB"], axis=1)
    tdf["qC"] = df.apply(lambda row: row["question"] + " " + row["answerC"], axis=1)
    tdf["qD"] = df.apply(lambda row: row["question"] + " " + row["answerD"], axis=1)
    tdf.index = df.index
    return tdf


def scoredf(tdf, scorer):
    sdf = pd.DataFrame()
    c = 0
    tt = ticktock("Scorer")
    tt.tick()
    for i, row in tdf.iterrows():
        c += 1
        if c % 10 == 0:
            tt.tock("%d/%d" % (c, tdf.shape[0])).tick()
            break
        sdf = sdf.append(scorerow(row, scorer))
    #sdf.index = tdf.index
    #sdf = tdf.apply(lambda row: scorerow(row, scorer), axis=1)
    return sdf


def scorerow(row, scorer):
    return row.apply(lambda x: scorer(x))


def elemscorer(idx):
    idx = idx
    def scorer(el):
        res = idx.search(el)
        if len(res) > 0:
            res = res[0]
            return res["score"]
        else:
            return 0.0
    return scorer


def choose(sdf): # sdf only has qA, qB, qC, qD and the right index (question id's)
    ret = pd.DataFrame()
    ret["correctAnswer"] = sdf.idxmax(axis=1)
    ret.index = sdf.index
    return ret


def evalu(pred, orig):
    return (pred["correctAnswer"] == orig["correctAnswer"]).sum(axis=0)*1. / orig.shape[0]


if __name__ == "__main__":
    idx = WikipediaIndex(dir="../../../data/wikipedia/pagesidx/")
    run(idx)