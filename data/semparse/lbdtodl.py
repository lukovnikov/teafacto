import re


inp = "(count $0 (and (major:t $0) (city:t $0) (loc:t $0 arizona:s)))"
#inp = "(argmax $0 (and (river:t $0) (loc:t $0 new_york:s)) (len:i $0))"
#inp = "(lambda $0 e (and (state:t $0) (next_to:t $0 michigan:s)))"
#inp = "(lambda $0 e (loc:t san_diego_ca:c $0))"
#inp = "(lambda $0 e (and (major:t $0) (city:t $0) (loc:t $0 delaware:s)))"
#inp = "(capital:c iowa:s)"
#inp = "(argmin $0 (and (place:t $0) (loc:t $0 louisiana:s)) (elevation:i $0))"
#inp = "(lambda $0 e (and (state:t $0) (loc:t (argmax $1 (and (mountain:t $1) (loc:t $1 usa:co)) (elevation:i $1)) $0)))"
#inp = "(lambda $0 e (and (state:t $0) (loc:t (argmax $1 (river:t $1) (len:i $1)) $0)))"
#inp = "(lambda $0 e (and (state:t $0) (loc:t (argmin $1 (place:t $1) (elevation:i $1)) $0)))"
# !!! wrong, no error message inp = "(count $0 (and (state:t $0) (next_to:t $0 (argmax $1 (state:t $1) (count $2 (and (state:t $2) (next_to:t $1 $2)))))))"
inp = "(capital:c (argmax $1 (state:t $1) (count $2 (and (state:t $2) (next_to:t $1 $2)))))"
#inp = "(argmax $0 (and (river:t $0) (loc:t $0 (and (state:t $1) (loc:t (argmax $2 (place:t $2) (elevation:i $2)) $1)))) (len:i $0))"
#inp = "(lambda $0 e (and (state:t $0) (loc:t mississippi_river:r $0) (loc:t (argmin $1 (and (place:t $1) (exists $2 (and (state:t $2) (loc:t mississippi_river:r $2) (loc:t $1 $2)))) (elevation:i $1)) $0)))"
#inp = "(count $0 (and (state:t $0) (exists $1 (and (city:t $1) (named:t $1 rochester:n) (loc:t $1 $0)))))"
#inp = "(lambda $0 e (and (state:t $0) (not (loc:t mississippi_river:r $0))))"
#inp = "(population:i (capital:c georgia:s))"
inp = "(argmax $0 (and (city:t $0) (loc:t $0 usa:co)) (size:i $0))"
inp = "(argmax $0 (and (river:t $0) (loc:t $0 (argmin $1 (and (state:t $1) (loc:t $1 usa:co)) (size:i $1)))) (len:i $0))"
inp = "(count $0 (and (major:t $0) (city:t $0) (exists $1 (and (state:t $1) (next_to:t $1 utah:s) (loc:t $0 $1)))))"

def transform(x, stack):
    x = re.sub("\)", " ) ", x)
    x = re.sub("\(", " ( ", x)
    x = re.sub("\s{2,}", " ", x)
    x = re.sub("^\s", "", x)
    x = re.sub("\s$", "", x)
    #print x
    # find arguments
    args = []
    i = 2
    previ = i
    nest = 0
    while i < len(x):
        if x[i] == " " and nest == 0:
            args.append(x[previ:i])
            previ = i + 1
        elif x[i] == "(":
            nest += 1
        elif x[i] == ")":
            nest -= 1
        i += 1
    if len(args) == 0:      # individual case
        if x[0] == "$":
            raise Exception()
        else:
            return x
    elif args[0] == "lambda":
        stack.append((args[0], args[1]))
        ret = "{}".format(transform(args[3], stack))
        stack.pop()
        return ret
    elif args[0] == "exists":
        stack.append((args[0], args[1]))
        ret = transform(args[2], stack)
        s = stack.pop()
        if len(s) == 3:
            if s[2][1] == args[1]:
                ret = "( {} . {} )".format(ret, s[2][0])
            elif s[2][2] == args[1]:
                ret = "( {} . {} )".format(s[2][0], ret)
            else:
                raise Exception()
        return ret
    elif args[0] == "count":
        stack.append((args[0], args[1]))
        ret = "( {} . count )".format(transform(args[2], stack))
        stack.pop()
        return ret
    elif args[0] == "not":
        stack.append((args[0],))
        ret = "( not {} )".format(transform(args[1], stack))
        stack.pop()
        return ret
    elif args[0] == "argmax" or args[0] == "argmin":
        stack.append((args[0], args[1]))
        splits = args[3].split()
        if len(splits) == 4:
            l = splits[1]
            r = splits[2]
            if r != args[1]:
                raise Exception("wrong $")
        else:
            l = transform(args[3], stack)
        ret = "( {} . {} . ( $ . {} ) )".format(transform(args[2], stack), args[0], l)
        stack.pop()
        return ret
    elif args[0] == "and":
        stack.append((args[0],))
        ts = [transform(x, stack) for x in args[1:]]
        ret = "( {} )".format(" and ".join(map(str,
                            filter(lambda x: x is not None, ts))))
        stack.pop()
        return ret
    elif len(args) == 2 and re.match("[\w_]+:[\w_]{1,2}", args[0]) and re.match("[\w_]+:[\w_]{1,2}", args[1]):
        return "( {} . {} )".format(args[1], args[0])
    elif re.match("[\w_]+:[\w_]", args[0]) and len(args) == 2:
        if args[1][0] == "$":       # type
            return "( {} )".format(args[0])
        else:                       # relation
            stack.append((args[0],))
            ret = "( {} . {} )".format(transform(args[1], stack), args[0])
            stack.pop()
            return ret
    elif re.match("[\w_]+:[\w_]", args[0]) and len(args) == 3:  # relation
        stack.append((args[0],))
        ret = None
        if args[1][0] == "$" and args[2][0] == "$":     # backref
            backref = args
            i = len(stack)
            while i > 0:
                i -= 1
                if stack[i][0] == "exists":
                    stack[i] = stack[i] + (tuple(backref),)
                    break
            ret = None
        elif args[1][0] == "$":       # var first
            ret =  "( {} . {} )".format(args[0], transform(args[2], stack))
        elif args[2][0] == "$":
            ret = "( {} . {} )".format(transform(args[1], stack), args[0])
        stack.pop()
        return ret
    return args


if __name__ == "__main__":
    print inp
    print " ========> "
    print transform(inp, [])