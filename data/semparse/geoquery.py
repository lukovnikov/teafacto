import re


def parse(inp="geoquery.pl", outp="geoquery.txt"):
    ''' parses Prolog files from Geo880 to text format for seq2seq'''
    pre = re.compile("parse\(\[(.+)\], (.+)\)\.\\n")
    with open(outp, "w") as f:
        for line in open(inp):
            gs = pre.match(line).groups()
            q = " ".join(gs[0].replace("'.'", ".").split(","))
            a = gs[1]
            a = parseanswer(a)
            f.write("{}\t{}\n".format(q, a))

def parseanswer(x):
    ret = []
    acc = ""
    brackre = re.compile("(.+)'([\w\s]+)'(.+)")
    m = brackre.match(x)
    if m:
        x = m.group(1) + m.group(2).replace(" ", "-") + m.group(3)
    x = x.replace("\+ ", "+").replace("\+", "+")
    for c in x:
        if re.match("[\w_-]+", c):
            if re.match("[\w_-]+", acc):     # append current character
                acc += c
            else:
                if len(acc) > 0:
                    ret.append(acc)
                acc = c
        elif c in set("( ) , +".split()):
            if len(acc) > 0:
                ret.append(acc)
            acc = c
    if len(acc) > 0:
        ret.append(acc)
    ret = " ".join(ret)
    try:
        assert(ret.replace(" ", "") == x)
    except AssertionError, e:
        xqdsf = ret.replace(" ", "")
        print ret
        raise e
    return ret



if __name__ == "__main__":
    parse()
