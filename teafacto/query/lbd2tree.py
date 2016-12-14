import re, numpy as np

### ALL OF THESE ARE NOT USED ###
class Node(object):
    ''' superclass for all nodes in parse tree'''
    _schema = None

    def __init__(self, *children):
        self._children = []  # nodes
        self._parent = None
        self.value = None

        self.children = children

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children):
        if self._schema != None:
            i = 0
            for selem in self._schema:
                if isinstance(selem, tuple):
                    assert(len(selem) == 3)
                    minc = selem[0]
                    maxc = selem[1] if selem[1] is not None else np.infty
                    typ = selem[2]
                    j = 0
                    while i + j < len(children) and j < maxc:
                        if not isinstance(children[i+j], typ):
                            raise Exception("wrong arg node")
                        j += 1
                    if j <= minc - 1:
                        raise Exception("more expected")
                    if maxc is not None and j > maxc:
                        j = maxc
                    i += j
                elif issubclass(selem, Node):
                    if isinstance(children[i], selem):
                        i += 1
                    else:
                        raise Exception("wrong arg node type")
            if i < len(children):
                raise Exception("too many arguments")
        self._children = children
        for child in children:
            child.parent = self

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, p):
        self._parent = p

    @property
    def haschildren(self):
        return len(self.children) > 0

    @property
    def printname(self):
        return self.value

    def __str__(self):
        if self.printname is not None:
            if self.haschildren:
                return "( {} {} )".format(self.printname, " ".join(map(str, self.children)))
            else:
                return "{}".format(self.printname)
        else:
            return "Node({})".format(" ".join(map(str, self._children)))

    def __repr__(self):
        return str(self)


class AtomicNode(Node):
    _schema = []


class SelectorNode(Node):
    ''' Node representing an assertion. Atomic or Type or Relation assertion '''
    pass


class VariableNode(AtomicNode, SelectorNode):
    ''' Node representing a variable. Has no children '''

    def __init__(self, id):
        super(VariableNode, self).__init__()
        self.value = id

    def __str__(self):
        return "${}".format(self.value)


class EntityNode(AtomicNode, SelectorNode):
    ''' Node representing an entity. Has no children '''

    def __init__(self, name):
        super(EntityNode, self).__init__()
        self.value = name


class TypeNode(SelectorNode):
    ''' Node representing a type assertion. Has one child '''
    _schema = [VariableNode]

    def __init__(self, typeid, child):
        super(TypeNode, self).__init__(child)
        self.value = typeid


class RelationNode(SelectorNode):
    ''' Node representing a relation assertion. Has two children '''
    _schema = [SelectorNode, SelectorNode]

    def __init__(self, relid, *children):
        super(RelationNode, self).__init__(*children)
        self.value = relid


class LambdaNode(Node):
    ''' Node representing a lambda. Has two children, first variable node, second expression '''
    _schema = [VariableNode, SelectorNode]


class ExistsNode(SelectorNode):
    _schema = [VariableNode, SelectorNode]


class PropertyNode(Node):
    _schema = [VariableNode]


class ArgOptNode(SelectorNode):
    ''' Node representing an argmax or an argmin. Has three children, var node, selector, criterion. '''
    _schema = [VariableNode, SelectorNode, PropertyNode]

    def __init__(self, kind, *children):
        super(ArgOptNode, self).__init__(*children)
        self.value = kind


class ArgmaxNode(ArgOptNode):
    def __init__(self, *children):
        super(ArgmaxNode, self).__init__("argmax", *children)


class ArgminNode(ArgOptNode):
    def __init__(self, *children):
        super(ArgminNode, self).__init__("argmin", *children)


class SetOpNode(SelectorNode):
    _schema = [(2, None, SelectorNode)]

    def __init__(self, op, *children):
        super(SetOpNode, self).__init__(*children)
        self.value = op


class AndNode(SetOpNode):
    def __init__(self, *children):
        super(AndNode, self).__init__("and", *children)


class OrNode(SetOpNode):
    def __init__(self, *children):
        super(OrNode, self).__init__("or", *children)


class CountNode(PropertyNode):
    _schema = [VariableNode, SelectorNode]

    @property
    def printname(self):
        return "count"


### WHAT IS ACTUALLY USED STARTS HERE ###
class ParseNode(object):
    def __init__(self, name, *children):
        self.children = children
        self.name = name

    @classmethod
    def create(cls, name, *children):
        if name == "and" and len(children) > 2:     # split in multiple ands
            cur = ParseNode(name, *children[:2])
            return ParseNode.create(name, *(cur,)+children[2:])
        else:
            return ParseNode(name, *children)

    @property
    def haschildren(self):
        return len(self.children) > 0

    @property
    def isvar(self):
        return re.match("\$.+", self.name) and not self.haschildren

    def __str__(self):
        if len(self.children) == 0:
            return "{}".format(self.name)
        else:
            return "( {} {} )".format(self.name, " ".join(map(str, self.children)))

    def __repr__(self):
        return str(self)

    @property
    def depth(self):
        maxchilddepth = reduce(max, map(lambda x: x.depth, self.children), -1)
        return maxchilddepth + 1

    def linearize(self, mode="greedy"):     # "deep" or "greedy"
        if mode == "greedy":
            return self.greedy_linearize_rec()
        elif mode == "deep":
            return self.deep_linearize()

    def deep_linearize(self):
        if len(self.children) == 0:
            return self.name, 0
        elif len(self.children) == 1:
            cl, d = self.children[0].deep_linearize()
            return "{} {} red-2".format(cl, self.name), d + 1
        elif self.name == "lambda":
            assert(len(self.children) == 3)
            fc, d = self.children[2].deep_linearize()
            return "{} {} {} red-3".format(fc,
                                     self.children[0].deep_linearize()[0],
                                     self.name), d + 1
        elif self.name == "exists" or self.name == "count":
            assert(len(self.children) == 2)
            fc, d = self.children[1].deep_linearize()
            return "{} {} {} red-3".format(fc,
                                  self.children[0].deep_linearize()[0],
                                  self.name), d + 1
        elif self.name == "argmax" or self.name == "argmin":
            assert(len(self.children) == 3)
            fc1, d1 = self.children[1].deep_linearize()
            fc2, d2 = self.children[2].deep_linearize()
            sign, f1, f2 = "+", fc1, fc2
            if d1 < d2:
                sign, f1, f2 = "-", fc2, fc1
            return "{} {} {} {}{} red-4".format(f1, fc2,
                                                self.children[0].deep_linearize()[0],
                                                sign, self.name), max(d1, d2) + 1
        elif self.name == "and":
            assert(len(self.children) == 2)
            fc1, d1 = self.children[0].deep_linearize()
            fc2, d2 = self.children[1].deep_linearize()
            lc, rc = fc1, fc2
            if d1 < d2:
                lc, rc = fc2, fc1
            return "{} {} {} red-3".format(lc, rc, self.name), max(d1, d2) + 1
        elif len(self.children) == 2:
            fc1, d1 = self.children[0].deep_linearize()
            fc2, d2 = self.children[1].deep_linearize()
            sign, lc, rc = "+", fc1, fc2
            if d1 < d2:
                sign, lc, rc = "-", fc2, fc1
            return "{} {} {}{} red-3".format(lc, rc, sign, self.name), max(d1, d2) + 1
        else:
            raise Exception("too many arguments")

    def greedy_linearize(self, deeppref=False):
        ret = self.greedy_linearize_rec(deeppref=deeppref)[0]
        ret = self.rename_vars(ret)
        return ret

    def rename_vars(self, x):
        xs = x.split()
        varstrings = filter(lambda a: re.match("\$\d", a), xs)
        vars = []
        for varstring in varstrings:
            if varstring not in vars:
                vars.append(varstring)
        varmap = zip(vars,
                     map(lambda x: "$-{}".format(x), range(len(varstrings))))
        for f, t in varmap:
            x = x.replace(f, t)
        x = x.replace("$-", "$")
        return x


    def greedy_linearize_rec(self, deeppref=False):
        if len(self.children) == 0:
            return self.name, 0
        elif len(self.children) == 1:
            cl, d = self.children[0].greedy_linearize_rec(deeppref=deeppref)
            return "{} {} red-2".format(cl, self.name), d
        elif self.name == "lambda":
            assert(len(self.children) == 3)
            fl, dl = self.children[2].greedy_linearize_rec(deeppref=deeppref)
            fr, dr = self.children[0].greedy_linearize_rec(deeppref=deeppref)
            return "{} {} {} red-3".format(fl, fr, self.name), max(dl, dr) + 1
        elif self.name == "exists" or self.name == "count":
            assert(len(self.children) == 2)
            fl, dl = self.children[1].greedy_linearize_rec(deeppref=deeppref)
            fr, dr = self.children[0].greedy_linearize_rec(deeppref=deeppref)
            return "{} {} {} red-3".format(fl, fr, self.name), max(dl, dr) + 1
        elif self.name == "argmax" or self.name == "argmin" or self.name == "sum":
            assert(len(self.children) == 3)
            fl, dl = self.children[1].greedy_linearize_rec(deeppref=deeppref)
            fm, dm = self.children[2].greedy_linearize_rec(deeppref=deeppref)
            fr, dr = self.children[0].greedy_linearize_rec(deeppref=deeppref)
            return "{} {} {} {} red-4".format(fl, fm, fr, self.name), max(dl, dm, dr) + 1
        elif self.name == "and" or self.name == "or":
            assert(len(self.children) == 2)
            fl, dl = self.children[0].greedy_linearize_rec(deeppref=deeppref)
            fr, dr = self.children[1].greedy_linearize_rec(deeppref=deeppref)
            if dl <= dr and deeppref:
                temp = fl
                fl = fr
                fr = temp
            return "{} {} {} red-3".format(fl, fr, self.name), max(dl, dr) + 1
        elif len(self.children) == 2:
            fl, dl = self.children[0].greedy_linearize_rec(deeppref=deeppref)
            fr, dr = self.children[1].greedy_linearize_rec(deeppref=deeppref)
            return "{} {} {} red-3".format(fl, fr, self.name), max(dl, dr) + 1
        else:
            print self
            raise Exception("too many arguments")

    def lambda_linearize(self):
        return str(self)


class LambdaParser(object):
    def __init__(self):
        self.stack = []

    def parse(self, s):
        i = 0
        j = 0
        while j < len(s):
            if s[j] in "( )".split():
                if i < j:
                    self.push(s[i:j])
                self.push(s[j])
                j += 1
                i = j
            elif s[j] == " ":
                if i < j:
                    self.push(s[i:j])
                j += 1
                i = j
            else:
                j += 1
        assert(len(self.stack) == 1)
        ret = self.stack[0]
        self.stack = []
        return ret

    def push(self, x):
        self.stack.append(x)
        self.reduce()

    def reduce(self):
        if self.stack[-1] == "(":      # do nothing
            pass
        elif self.stack[-1] != ")":        # try atomic reduce
            self.stack[-1] = ParseNode.create(self.stack[-1])
        else:
            j = 1       # find the matching "("
            while self.stack[-j] != "(":
                j += 1
            args = self.stack[-j+1:-1]      # find args
            self.stack = self.stack[:-(len(args)+2)]
            n = ParseNode.create(args[0].name, *args[1:])
            self.stack.append(n)


class GreedyLinParser(object):
    def __init__(self):
        self.stack = []

    def parse(self, x):
        xs = x.split()
        for xse in xs:
            self.push(xse)
        assert(len(self.stack) == 1)
        return self.stack[0]

    def push(self, s):
        self.stack.append(s)
        self.reduce()

    def reduce(self):
        m = re.match("red-(\d)", self.stack[-1])
        if m:
            self.stack.pop()
            numargs = int(m.group(1))
            args = self.stack[-numargs:]
            name = args[-1].name
            args = args[:-1]
            if name == "argmax" or name == "argmin":
                args = [args[2], args[0], args[1]]
            elif name == "lambda":
                args = [args[1], "e", args[0]]
            elif name == "count" or name == "exists":
                args = [args[1], args[0]]
            self.stack = self.stack[:-numargs]
            n = ParseNode.create(name, *args)
            self.stack.append(n)
        else:
            n = ParseNode.create(self.stack[-1])
            self.stack[-1] = n


if __name__ == "__main__":
    s = "(lambda $0 e (and (state:t $0) (> (elevation:i (argmax $1 (and (place:t $1) (loc:t $1 $0) (state:t $1)) (elevation:i $1))) (elevation:i (argmax $1 (and (place:t $1) (loc:t $1 colorado:s)) (elevation:i $1))))))"
    s = "(lambda $0 e (and (state:t $0) (> (elevation:i colorado:s) (elevation:i (argmax $1 (and (place:t $1) (loc:t $1 $0) (state:t $1)) (elevation:i $1))))))"
    #s = "(lambda $0 (and (state:t $0) (next_to:t $0 (argmax $1 (state:t $1) (major:t $1) (city:t $1) (population:i $1)))))"
    #s = "(count $0 (and (state:t $0) (loc:t mississippi_river:r $0)))"
    #s = "(count $0 ( state:t $0 ))"
    #s = "(lambda $0 (capital:t hawaii:s $0))"
    #s = "hawaii:s "
    #s = "(count $0 (and (state:t $0) (exists $1 (and (city:t $1) (named:t $1 rochester:n) (loc:t $1 $0)))))"
    #s = "(population:i (capital:t florida:s))"
    #s = "(lambda $0 e (and (river:t $0) (exists $1 (and (state:t $1) (next_to:t $1 new_mexico:s) (loc:t $0 $1)))))"
    #s = "(lambda $0 e (exists $1 (and (state:t $1) (next_to:t $1 mississippi:s) (high_point:t $1 $0))))"
    tree = LambdaParser().parse(s)
    print tree
    print tree.depth
    #print tree.deep_linearize()[0]
    gl = tree.greedy_linearize(deeppref=True)
    print gl
    tree = GreedyLinParser().parse(gl)
    print tree