import re, json
from IPython import embed
from teafacto.util import argprun


def inspect_data_raw(p="../../../data/WebQSP/data/"):
    traindatap = p+"WebQSP.train.json"
    trainrelations = []
    trainquestions = []
    relreg = re.compile("^(ns:(?![mng]\.).+)$")
    with open(traindatap) as f:
        d = json.load(f)["Questions"]
        for question in d:
            trainquestions.append(question["ProcessedQuestion"])
            for parse in question["Parses"]:
                sparql = parse["Sparql"]
                for s in sparql.split():
                    m = relreg.match(s)
                    if m:
                        trainrelations.append(m.group(1))
    testdatap = p+"WebQSP.test.json"
    testrelations = []
    testquestions = []
    with open(testdatap) as f:
        d = json.load(f)["Questions"]
        for question in d:
            testquestions.append(question["ProcessedQuestion"])
            for parse in question["Parses"]:
                sparql = parse["Sparql"]
                for s in sparql.split():
                    m = relreg.match(s)
                    if m:
                        testrelations.append(m.group(1))
    alluniquerelations = set(testrelations).union(set(trainrelations))
    allquestions = trainquestions.extend(testquestions)


    embed()


if __name__ == "__main__":
    argprun(inspect_data_raw)