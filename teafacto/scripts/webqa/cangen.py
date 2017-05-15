from teafacto.scripts.webqa.search import Searcher, EntityInfoGetter
import json, re, pickle
from IPython import embed
from teafacto.util import argprun, ticktock


def cangen_run(trainp="../../../data/WebQSP/data/WebQSP.train.json",
        testp="../../../data/WebQSP/data/WebQSP.test.json"):
    traind, alltrainmids = gen_and_save_cans(trainp)
    testd, alltestmids = gen_and_save_cans(testp)
    allmids = alltrainmids.union(alltestmids)
    print len(allmids)


def gen_and_save_cans(p, maxnumres=100):
    d = json.load(open(p))
    d = d["Questions"]
    search = Searcher()
    cangenrecall = 0.
    total = 0.
    c = 0
    allmids = set()
    for question in d:
        qstring = question["ProcessedQuestion"]
        cans = search.search_sentence(qstring,
            maxnumres=maxnumres, leastfuzzy=True)
        question["CandidateEntities"] = cans
        allmids.update(cans.keys())
        if question["Parses"][0]["TopicEntityMid"] in cans:
            cangenrecall += 1
        total += 1
        print "{:5d}: {:.4f} \t ({:6d})".format(c, cangenrecall / total, len(allmids))
        c += 1
    print "{}% questions found topic can".format(cangenrecall / total)
    embed()
    json.dump(d, open(p+".withcans", "w"))
    return d, allmids


def collect_and_save_mids_with_cans_run(trainp="../../../data/WebQSP/data/WebQSP.train.json.withcans",
        testp="../../../data/WebQSP/data/WebQSP.test.json.withcans",
        outp="../../../data/WebQSP/data/WebQSP.allmids.withcans.pkl"):
    sparql = SPARQLWrapper("http://drogon:9890/sparql")
    sparql.setReturnFormat(JSON)
    trainmids = collect_mids(trainp)
    testmids = collect_mids(testp)
    pickle.dump({"trainmids": trainmids, "testmids": testmids},
                open(outp, "w"))


def collect_mids(p):
    entre = re.compile("ns:([\d\w_\.]+)")
    mids = set()
    for question in json.load(open(p)):
        mids.update(question["CandidateEntities"].keys())
        for parse in question["Parses"]:
            sparqlwords = parse["Sparql"].split()
            entnames = map(lambda x: entre.match(x).group(1),
                           filter(lambda x: entre.match(x), sparqlwords))
            mids.update(set(entnames))
    print len(mids)
    return mids


def get_mids_info(p="../../../data/WebQSP/data/WebQSP.allmids.withcans.pkl",
                  outp="../../../data/WebQSP/data/WebQSP.allmids.withcans.info.pkl"):
    mids = pickle.load(open(p))
    eig = EntityInfoGetter()
    allmids = mids["trainmids"].union(mids["testmids"])
    allinfo = {}
    typemids = set()
    tt = ticktock("mid info getter")
    tt.tick()
    for mid in allmids:
        if len(allinfo) % 10 == 0:
            tt.tock("{:6}/{:6}".format(len(allinfo), len(allmids)))
            tt.tick()
        midinfo = eig.get_info(mid).values()[0]
        allinfo[mid] = midinfo
        typemids.update(set(midinfo.notable_types))
        typemids.update(set(midinfo.types))
        if len(allinfo) % 100 == 0:
            break
            pass
    print len(typemids)
    for mid in typemids:
        midinfo = eig.get_info(mid).values()[0]
        allinfo[mid] = midinfo
    todump = map(lambda x: x.finalize(), allinfo.values())
    pickle.dump(todump, open(outp, "w"))
    embed()


if __name__ == "__main__":
    argprun(get_mids_info)




