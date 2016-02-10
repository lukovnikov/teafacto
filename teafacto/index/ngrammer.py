import nltk, sys, os, re
import pandas as pd, numpy as np
from IPython import embed

df = pd.DataFrame.from_csv("alldata.csv")

print "loaded csv"

tsentence = df["question"].iloc[0]

print tsentence

numgrams = 3
wre = re.compile("^[\w\s]+$")

def ngrammer(inp):
	tokens = map(lambda x: x.lower(), nltk.wordpunct_tokenize(inp))
	#print tokens
	ngrams = []
	ngrams.extend(tokens)
	ngrams.extend([" ".join(x) for x in nltk.bigrams(tokens)])
	ngrams.extend([" ".join(x) for x in nltk.trigrams(tokens)])
	#print ngrams
	ret = filter(lambda x: wre.match(x), ngrams)
	return ret

print ngrammer(tsentence)

ngrams = set()

embed()
ndf = df[["question","answerA","answerB","answerC","answerD"]].applymap(ngrammer)
rndf = df.apply(lambda col: reduce(lambda x,y: x+y, col, 0))
rrn = reduce(lambda x,y: x.union(y), map(lambda x: set(x), rndf), set())
print rndf
print len(rrn)

c = 0
for i, row in df.iterrows():
	break
	c += 1
	if c % 1000 == 0:
		print c
	rowngrams = set()
	for x in ["question", "answerA", "answerB", "answerC", "answerD"]:
		rowngrams = rowngrams.union(set(ngrammer(row[x])))
	ngrams = ngrams.union(rowngrams)


print len(ngrams)

