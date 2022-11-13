import copy, math, fractions, decimal, operator

import pyparsing.common
from pyparsing import *
ParserElement.enable_packrat()

OP_REMAPS = { '**': '^', '~' : 'not', "|" : 'or', "&" : 'and'}
KEYWORDS = {'sin', 'cos', 'tan'}
FORBIDDEN = ["and", "or", "not"]
CONSTANT_NUMBERS = ["e", "pi"]
LATEX_UNARY = {
        "!" : lambda x : f"{x}!",
        "+" : lambda x : f"+{x}",
        "-" : lambda x : f"-{x}",
        }
LATEX_BINARY = {
        "^" : lambda x,y : f"{x}^{{{y}}}",
        "*" : lambda x,y : f"{x} \cdot {y}",
        "+" : lambda x,y : f"{x} + {y}",
        "-" : lambda x,y : f"{x} - {y}",
        "." : lambda x,y : f"{x} \circ {y}",
        "/" : lambda x,y : f"\\frac{{{x}}}{{{y}}}",
        }


def any_void(*args):
    return any(map(lambda x : isinstance(x, VoidNode), args))

def all_type(cl, *args):
    return all(map(lambda x : isinstance(x, cl), args))

def any_type(cl, *args):
    return any(map(lambda x : isinstance(x, cl), args))

def all_primitive(*args):
    return all_type(PrimitiveNode, *args)

def void_wrapper(func):
    return lambda *args : VoidNode() if any_void(*args) else func(*args)

def boolean_false():
    return BooleanNode(0)

def is_false(x):
    return isinstance(x, NumberNode) and x.contents == 0

def is_true(x):
    return isinstance(x, NumberNode) and x.contents == 1

def boolean_true():
    return BooleanNode(1)

def bool_wrapper(func):
    return lambda *args : boolean_true() if func(*args) else boolean_false()

def to_decimal(frac):
    if isinstance(frac, fractions.Fraction):
        return decimal.Decimal(frac.numerator) / decimal.Decimal(frac.denominator)
    else:
        return decimal.Decimal(frac)

def num_wrapper(func):
    def res(*args):
        if all_type(NumberNode, *args):
            if any_type(FloatNode, *args):
                newargs = map(lambda x : to_decimal(x.contents), args)
            else:
                newargs = map(lambda x : x.contents, args)
            ans = func(*newargs)
            if isinstance(ans, int):
                return IntegerNode(ans)
            elif isinstance(ans, fractions.Fraction):
                if ans.denominator == 1:
                    return IntegerNode(ans)
                else:
                    return RationalNode(ans.numerator, ans.denominator)
            else:
                return FloatNode(ans)
        else:
            return VoidNode()
    return res

def similar_wrapper(func):
    def res(*args):
        if all_type(NumberNode, *args):
            return (num_wrapper(func))(*args)
        elif all_type(LiteralNode, *args):
            return LiteralNode(func(*map(lambda x : x.contents, args)))
        else:
            return VoidNode()
    return res 

def lit_wrapper(func):
    return lambda *args : LiteralNode(func(*args))


class Node:

    def __init__(self):
        pass


    def __repr__(self):
        return f"Node (unknown)"


    def pprint(self,indent="|"):
        print(f"Node (unknown)")


class RefNode(Node):

    def __init__(self, name):
        super().__init__()
        self.name = name


    def __repr__(self):
        return f"Ref({self.name})"


    def pprint(self,indent="|"):
        addon = f"Ref({self.name})"
        print(indent+addon)


    def evaluate(self, scope=None, *args): 
        if scope is not None and scope.includes(self.name):
            fn = scope.get(self.name)
            #return scope.get(self.name)
            if isinstance(fn, FunctionNode):
                return fn
            else:
                return fn.evaluate(scope, *args)
        else:
            return self


    def latex(self):
        return self.name


class CallNode(Node):

    def __init__(self, call, args):
        super().__init__()
        self.call = call
        self.args = args


    def __repr__(self):
        return f"Call({self.call}, {self.args})"


    def pprint(self,indent="|"):
        print(indent+f"Call:")
        self.call.pprint("      " + indent)
        for a in self.args:
            a.pprint("      " + indent)
    

    def evaluate(self, scope=None, *args):
        newargs = list(map(lambda x : x.evaluate(scope), self.args))
        if all_primitive(*newargs):
            called = self.call.evaluate(scope)
            res = called.evaluate(scope, *newargs)
            if isinstance(res, RefNode):
                return CallNode(res, newargs)
            else:
                return res
        else:
            return self


    def latex(self):
        argstr = ", ".join(x.latex() for x in self.args)
        return f"{self.call}({argstr})"


class VarNode(Node):

    def __init__(self, name, expr_out):
        super().__init__()
        self.name = name
        self.expr_out = expr_out


    def __repr__(self):
        return f"Var({self.name} :  {self.expr_out})"


    def pprint(self, indent="|"):
        print(f"{indent}Var: {self.name}")
        self.expr_out.pprint(indent="      "+indent)


    def evaluate(self, scope=None):
        if scope is not None:
            expr_out = self.expr_out.evaluate(scope)
            scope.assign_var(self.name,expr_out)
        return self


    def latex(self):
        return self.expr_out.latex()


class DefNode(Node):

    def __init__(self, name, vars_in, expr_out):
        super().__init__()
        self.name = name
        self.vars_in = vars_in
        self.expr_out = expr_out


    def __repr__(self):
        return f"Define({self.name} : {self.vars_in} -> {self.expr_out})"


    def pprint(self, indent="|"):
        vout = ""
        for v in self.vars_in:
            vout += v.name
        vout = ", ".join(vout)
        print(f"{indent}Def: {self.name}")
        print(f"      {indent}Args: {vout}")
        self.expr_out.pprint(indent="      "+indent)


    def evaluate(self, scope=None):
        if scope is not None:
            scope.assign(self.name,self.vars_in,self.expr_out)
        return self


    def latex(self):
        return f"{self.name}({', '.join(self.args)}) = {self.expr_out.latex()}"


class FunctionNode(Node):

    def __init__(self, args, evalfn):
        super().__init__()
        self.args = args
        self.evalfn = evalfn
        self.arity = len(args)


    def __repr__(self):
        return f"Func({self.args} : {self.evalfn})"


    def pprint(self, indent="|"):
        #i = len(self.name)*" "
        vout = ""
        for v in self.args:
            vout += v.name
        vout = ", ".join(vout)
        print(f"{indent}Func: ")
        print(f"      {indent}Args: {vout}")
        self.evalfn.pprint(indent="      "+indent)


    def compatible(self, *args): 
        return len(args) == len(self.args)


    def evaluate(self, scope = None, *args):
        if self.compatible(*args):
            if len(self.args) == 0:
                return self.evalfn.evaluate(scope)
            elif len(args) == 0:
                return self
            else:
                newscope = Scope(scope)
                for a,v in zip(self.args,args):
                    ev = v.evaluate(scope, *args)
                    if not isinstance(ev, PrimitiveNode):
                        return self
                    newscope.assign_var(a.name, ev)
                return self.evalfn.evaluate(newscope)
        else:
            return VoidNode()
        

    def latex(self):
        argstr = ", ".join(x.latex() for x in self.args)
        return f"f({argstr}) = {self.evalfn.latex()}"


class BuiltinNode(FunctionNode):
    def __init__(self, name, arity, call):
        self.name = name
        self.arity = arity
        self.call = call

    def __repr__(self):
        return f"Builtin({self.arity} : {self.name})"

    def pprint(self, indent="|"):
        print(f"{indent}Builtin: ")
        print(f"        {indent}Arity: {self.arity}")
        print(f"        {indent}Name: {self.name}")


    def compatible(self, *args):
        return len(args) == self.arity

    
    def evaluate(self, scope = None, *args):
        if self.compatible(*args):
            return self.call(*args)
        else:
            return VoidNode()


    def latex(self):
        return f"{self.name}"


class PrimitiveNode(FunctionNode):

    def __init__(self, contents):
        super().__init__([], self)
        self.contents = contents
        self.TERNARY = {
                "if": void_wrapper(lambda x,y,z : z if is_false(x) else y),
                }
        self.BINARY = {
                ".": void_wrapper(lambda x,y : x), 
                "and": void_wrapper(lambda x,y : boolean_false() if is_false(x) else copy.copy(y)),
                "or": void_wrapper(lambda x,y : y if is_false(x) else x),
                "==": bool_wrapper(lambda x,y : x.contents == y.contents),
                "#": bool_wrapper(lambda x,y : x.contents != y.contents),
                "<": bool_wrapper(lambda x,y : x.contents < y.contents),
                "<=": bool_wrapper(lambda x,y : x.contents <= y.contents),
                ">": bool_wrapper(lambda x,y : x.contents > y.contents),
                ">=": bool_wrapper(lambda x,y : x.contents >= y.contents),
                }
        self.UNARY = {
                "not": void_wrapper(lambda x : boolean_true() if is_false(x) else boolean_false()),
                }


    def __repr__(self):
        return f"Primitive({self.contents})"


    def pprint(self,indent="|"):
        print(f"{indent}Primitive: {self.contents}")


    def compatible(self, *args):
        return True


    def binary_op(self, op, other):
        if op in self.BINARY:
            return self.BINARY[op](self, other)
        else:
            return VoidNode()


    def unary_op(self, op):
        if op in self.UNARY:
            return self.UNARY[op](self)
        else:
            return VoidNode()


    def ternary_op(self, op, other1, other2):
        if op in self.TERNARY:
            return self.TERNARY[op](self, other1, other2)
        else:
            return VoidNode()


    def operate(self, op, *args):
        if len(args) == 0:
            return self.unary_op(op)
        elif len(args) == 1:
            return self.binary_op(op, args[0])
        elif len(args) == 2:
            return self.ternary_op(op, args[0], args[1])
        else:
            return VoidNode()


    def evaluate(self, scope=None, *args):
        return self


    def latex(self):
        return self.contents


class ListNode(PrimitiveNode):
    
    def __init__(self, elements):
        super().__init__(elements)
        self.elements = elements
        self.BINARY.update({ "get": lambda x,y: x.get_element(y) })


    def __repr__(self):
        return f"List({self.elements})"


    def pprint(self, indent="|"):
        print(f"{indent}List:")
        for a in self.elements:
            a.pprint("      " + indent)


    def get_element(self, index_node):
        if isinstance(index_node, IntegerNode) and index_node.contents < len(self.elements):
            return self.elements[int(index_node.contents)]
        else:
            return VoidNode()


    def evaluate(self, scope=None, *args):
        evals = [x.evaluate(scope, *args) for x in self.elements]
        return ListNode(evals)


    def latex(self):
        els = ", ".join([x.latex() for x in self.elements])
        return f'[{els}]'

class NumberNode(PrimitiveNode):

    def __init__(self, contents):
        super().__init__(contents)
        self.BINARY.update({
            "+": num_wrapper(operator.add),
            "-": num_wrapper(operator.sub),
            "*": num_wrapper(operator.mul),
            "/": num_wrapper(operator.truediv),
            "^": num_wrapper(operator.pow),
            })
        self.UNARY.update({
            "-": num_wrapper(operator.neg), 
            "+": num_wrapper(lambda x: x), 
            })

    def __repr__(self):
        return f"Number({self.contents})"


    def pprint(self,indent="|"):
        print(f"{indent}Number: {self.contents}")


class RationalNode(NumberNode):

    def __init__(self, numerator, denominator):
        super().__init__(fractions.Fraction(numerator, denominator))


    def __repr__(self):
        return f"Rational({self.contents})"


    def pprint(self,indent="|"):
        print(f"{indent}Rational: {self.contents}")


    def latex(self):
        return f"\\frac{{{self.contents.numerator}}}{{{self.contents.denominator}}}"


class IntegerNode(RationalNode):

    def __init__(self, contents):
        super().__init__(contents, 1)
        self.UNARY.update({ "!": num_wrapper(lambda x : math.factorial(x.numerator)), })


    def __repr__(self):
        return f"Int({self.contents.numerator})"


    def pprint(self,indent="|"):
        print(f"{indent}Int: {self.contents.numerator}")


    def latex(self):
        return f"{self.contents.numerator}"


class BooleanNode(IntegerNode):

    def __init__(self, contents):
        super().__init__(contents)


    def __repr__(self):
        return f"Bool({self.contents.numerator})"


    def pprint(self,indent="|"):
        print(f"{indent}Bool: {self.contents.numerator}")


    def latex(self):
        return f"{self.contents.numerator}"


class FloatNode(NumberNode):

    def __init__(self, contents):
        super().__init__(contents)


    def __repr__(self):
        return f"Float({self.contents})"


    def pprint(self,indent="|"):
        print(f"{indent}Float: {self.contents}")


    def latex(self, dp=None):
        if dp is not None:
            return f"{round(self.contents,dp)}"
        else:
            return f"{self.contents}"

class LiteralNode(PrimitiveNode):

    def __init__(self, contents):
        super().__init__(str(contents))


    def __repr__(self):
        return f"Literal({self.contents})"


    def pprint(self,indent="|"):
        print(f"{indent}Literal: {self.contents}")


    def latex(self):
        return self.contents


class VoidNode(PrimitiveNode):

    def __init__(self):
        super().__init__(None)


    def __repr__(self):
        return f"Void()"


    def pprint(self,indent="|"):
        print(f"{indent}Void: Void")


    def binary_op(self, op, other):
        return VoidNode()


    def unary_op(self, op, other):
        return VoidNode()


    def latex(self):
        return "\\operatorname{Void}"


class OpNode(Node):

    def __init__(self, call, args, remaps=OP_REMAPS):
        super().__init__()
        self.call = remaps.get(call, call)
        self.args = args


    def __repr__(self):
        return f"Op({self.call} : {self.args})"


    def pprint(self,indent="|"):
        print(f"{indent}Op: {self.call}")
        for a in self.args:
            a.pprint("     "+indent)


    def evaluate(self, scope=None, *args):
        subargs = list(map(lambda x : x.evaluate(scope, *args), self.args))
        if self.call == 'list':
            return ListNode(subargs)
        elif all_primitive(*subargs):
            if len(subargs) == 1:
                return subargs[0].unary_op(self.call)
            elif len(subargs) == 2:
                return subargs[0].binary_op(self.call, subargs[1])
            elif len(subargs) == 3:
                return subargs[0].ternary_op(self.call, subargs[1], subargs[2])
            else:
                return VoidNode()
        else:
            return OpNode(self.call, subargs)


    def latex(self):
        def texcheck(x):
            return f"\\left({x.latex()}\\right)"if isinstance(x, OpNode) else x.latex()
        if len(self.args) == 1:
            call = LATEX_UNARY[self.call]
        elif len(self.args) == 2:
            call = LATEX_BINARY[self.call]
        elif len(self.args) == 3:
            call = LATEX_TERNARY[self.call]
        else:
            return VoidNode().latex()
        return call(*map(texcheck, self.args))


def parse_void(tokens):
    return VoidNode()


def parse_symbol(tokens):
    return RefNode(tokens[0])


def parse_list(tokens):
    return OpNode('list', tokens)


def parse_integer(tokens):
    return IntegerNode(int(tokens[0]))


def parse_real(tokens):
    ans = decimal.Decimal(".".join(tokens))
    return FloatNode(ans)


def parse_prefix(tokens):
    if len(tokens[0]) == 0:
        return tokens[1]
    else:
        res = tokens[1]
        for op in reversed(tokens[0]):
            res = OpNode(op, [res])
        return res


def parse_postfix(tokens):
    if len(tokens[1]) == 0:
        return tokens[0]
    else:
        res = tokens[0]
        for op in tokens[1]:
            if isinstance(op, list):
                if len(op) == 2 and op[0] in {'eval','get'}:
                    kind,ind = op 
                    if kind == 'eval':
                        res = CallNode(res, list(ind))
                    else:
                        res = OpNode('get', [res, ind])
                else:
                    raise Exception("this shouldn't happen lol")
            else:
                res = OpNode(op, [res])
        return res


def parse_left_assoc(tokens):
    if len(tokens) == 1:
        return tokens[0]
    else:
        tokens = tokens.copy()
        left = tokens.pop(0)
        while len(tokens) > 0:
            op = tokens.pop(0)
            right = tokens.pop(0)
            left = OpNode(op, [left, right])
        return left


def parse_right_assoc(tokens):
    if len(tokens) == 1:
        return tokens[0]
    else:
        tokens = tokens.copy()
        right = tokens.pop()
        while len(tokens) > 0:
            op = tokens.pop()
            left = tokens.pop()
            right = OpNode(op, [left, right])
        return right


def parse_literal(tokens):
    tkns = tokens
    return LiteralNode(tkns[0])


def parse_ifthenelse(tokens):
    if len(tokens) == 1:
        return tokens[0]
    else:
        return OpNode("if", [tokens[0], tokens[1], tokens[2]])


def parse_def_fun(tokens):
    return DefNode(tokens[0].name, list(tokens[1]), tokens[2])


def parse_def_var(tokens):
    if len(tokens) == 2:
        return VarNode(tokens[0].name, tokens[1])
    else:
        return DefNode(tokens[0].name, list(tokens[1]), tokens[2])


def grammar(forbidden = FORBIDDEN, constant_nums = CONSTANT_NUMBERS, void="Void"): 
    left, right, leftsq, rightsq = map(Suppress, "()[]")

    void_symbol = Keyword(void)
    keyword = Or(map(Keyword,forbidden + [void]))
    constant_number = Or(map(Keyword, constant_nums))

    op_exp = one_of("^ **")
    op_sign = one_of("+ -")
    op_prod = one_of("* /")
    op_add = one_of("+ -")
    op_comp = Literal(".")
    op_rel = one_of("== < <= > >= !=")
    op_factorial = ~Literal("!=") + Literal("!") 
    op_and = Keyword("and")
    op_or = Keyword("or")
    op_not = Keyword("not") | Literal("~")
    op_if = Suppress("=>")
    op_else = Suppress(Keyword("else"))
    op_prefix = op_sign | op_not

    expression = Forward()
    ifthen_term = Forward()
    not_term = Forward()
    and_term = Forward()
    or_term = Forward()
    rel_term = Forward()
    sum_term = Forward()
    prod_term = Forward()
    pow_term = Forward()

    #symbol = Word(alphas, alphanums + "_")
    symbol = (~constant_number + ~keyword + Word(alphas, alphanums + "_"))
    integer = Word(nums)
    real = Combine(Word(nums) + "." + Word(nums))
    string = QuotedString('"', escChar="\\") | QuotedString("'", escChar="\\")
    number = real | integer
    var_args = delimited_list(symbol)
    list_def = leftsq + delimited_list(expression) + rightsq

    atom = (symbol | number | string | list_def | left + expression + right)
    args = left + Opt(delimited_list(expression)) + right
    sqbracket = leftsq + expression + rightsq
    postfix = atom + Group((op_factorial | sqbracket | args)[...])

    prefix = Group(op_prefix[...]) + postfix
    pow_term <<= (prefix + op_exp + pow_term) | prefix
    prod_term <<= (pow_term + (op_prod + pow_term)[...])
    sum_term <<= (prod_term + (op_add + prod_term)[...])
    rel_term <<= (sum_term + (op_rel + rel_term)[...]) 
    and_term <<= (rel_term + (op_and + rel_term)[...]) 
    or_term <<= (and_term + (op_or + and_term)[...])
    op_ifmid = op_if + expression + op_else
    ifthen_term <<= (or_term + op_ifmid + expression) | or_term

    def_symbols = list(map(Suppress, [":", "->"]))
    def_fun = symbol + def_symbols[0] + Group(Opt(var_args)) + def_symbols[1] + expression
    def_var = symbol + def_symbols[0] + expression

    expression <<= def_fun | def_var | ifthen_term

    void_symbol.set_parse_action(parse_void)
    args.set_parse_action(lambda x : [['eval', list(x)]])
    sqbracket.set_parse_action(lambda x : [['get', x[0]]])
    symbol.set_parse_action(parse_symbol)
    integer.set_parse_action(parse_integer)
    real.set_parse_action(parse_real)
    string.set_parse_action(parse_literal)
    list_def.set_parse_action(parse_list)
    prefix.set_parse_action(parse_prefix)
    postfix.set_parse_action(parse_postfix)
    pow_term.set_parse_action(parse_right_assoc)
    prod_term.set_parse_action(parse_left_assoc)
    rel_term.set_parse_action(parse_left_assoc)
    sum_term.set_parse_action(parse_left_assoc)
    and_term.set_parse_action(parse_left_assoc)
    or_term.set_parse_action(parse_left_assoc)
    ifthen_term.set_parse_action(parse_ifthenelse)
    def_fun.set_parse_action(parse_def_fun)
    def_var.set_parse_action(parse_def_var)

    return expression


class SyntaxTree:

    def __init__(self, string):
        self.string = string
        self.tree = grammar().parse_string(self.string, parse_all=True)[0]
        #self.scope = Scope() if scope is None else scope


    def evaluate(self, scope=None):
        return self.tree.evaluate(scope)


    def pprint(self):
        self.tree.pprint()


class Scope:

    def __init__(self, parent=None):
        self.parent = parent
        self.assigned = dict()


    def pprint(self, indent="|"):
        for k,v in self.assigned.items():
            print(f"{k} = {v}")
        if self.parent is not None:
            print(indent + "parent:")
            self.parent.pprint("     " + indent)


    def assign(self, ref, args, evalfn):
        if ref not in FORBIDDEN:
            self.assigned[ref] = FunctionNode(args, evalfn)
            return self.assigned[ref]
        else:
            raise TypeError('bad assignment?')

    def assign_var(self, ref, evalfn):
        if ref not in FORBIDDEN:
            self.assigned[ref] = evalfn
            return self.assigned[ref]
        else:
            raise TypeError('bad assignment?')

    def get(self, ref):
        if ref in self.assigned:
            return self.assigned[ref]
        elif self.parent is not None:
            return self.parent.get(ref)
        else: 
            return ref


    def get_local(self, ref):
        if ref in self.assigned:
            return self.assigned[ref]
        else:
            return ref


    def includes(self, ref):
        if ref in self.assigned:
            return True
        elif self.parent is not None:
            return self.parent.includes(ref)
        else:
            return False


    def add_builtin(self, name, arity, call):
        self.assigned[name] = BuiltinNode(name, arity, call)


    def get_subs(self,digits):
        res = dict()
        for k,v in self.assigned.items():
            if not isinstance(v, BuiltinNode):
                if isinstance(v, FloatNode):
                    res[k] = v.latex(digits)
                else:
                    res[k] = v.latex()
        return res


class Environment:
    def round_fun(num,digits):
        if isinstance(num, NumberNode) and isinstance(digits, RationalNode):
            return FloatNode(round(to_decimal(num.contents), digits.contents.numerator))
        else:
            return VoidNode()

    def __init__(self):
        self.scope = Scope()
        self.scope.add_builtin('sqrt', 1, num_wrapper(math.sqrt))
        self.scope.add_builtin('min', 2, similar_wrapper(min))
        self.scope.add_builtin('max', 2, similar_wrapper(max))
        self.scope.add_builtin('exp', 1, num_wrapper(math.exp))
        self.scope.add_builtin('round', 2, Environment.round_fun)
        self.scope.add_builtin('numerator', 1, num_wrapper(lambda x: x.numerator))
        self.scope.add_builtin('denominator', 1, num_wrapper(lambda x: x.denominator))

    def process(self, string):
        tree = SyntaxTree(string)
        result = tree.evaluate(self.scope)
        return result


    def get(self, var):
        return SyntaxTree(var).evaluate(self.scope)


    def process_literally(self, var, res):
        try:
            string = f"{var} : {res}"
            tree = SyntaxTree(string)
            tree2 = SyntaxTree(var)
            tempscope = Scope()
            result = tree.evaluate(tempscope)
            if isinstance(tree2.evaluate(tempscope), NumberNode):
                self.process(string)
            else:
                self.process(f"{var} : '{res}'")
        except:
            self.process(f"{var} : '{res}'")


    def get_subs(self, digits=None):
        return self.scope.get_subs(digits)


if __name__ == "__main__":
    grammar().run_tests("""
                        -1
                        --1
                        x!
                        x[1]!
                        x[1]
                        x(1)
                        x(1,2)
                         x-y
                         x+y
                         x-y-z
                         x => (x+y) else z
                         x => (x+y) else z => q else z
                         x => (x+y) => r else z else z
                         x == y 
                         x-y
                         x >= y 
                         -1
                         -x
                         (x => (x+y) else z)  => r else z 
                         x^y^z
                         (x^y)^z
                         sin(x)
                         sin(x,y)
                         sin(x,y)!
                         sin(x)[x]
                         [1,2,3]
                         x[1]!
                         -x![1]
                         x: y
                         x : 3
                         x: 1+3
                         f: x,y -> x^2+y^2
                         f: x,y -> (+not -[y,x,3][1](1))
                         sin(f(x))
                         x[sin(f(-x!)!)!]
                         -+-+--+-(x+y)
                         x[1]
                         x!
                         x^y
                         x^y^z
                         (x^y)^z
                         (x+y)!
                         (x+y)![1]
                         (x+y)![1](2)
                         x+y[1]
                         x+y[(x-y)!+3]
                         x+y+z+w
                         x : 3*x+3
                         f : x,y -> 3*x+y
                         f : x,y -> g : a,b -> a+b
                         f : x,y -> (g : a,b -> a+b)(x,y)
                         x == y
                         x != y
                         x! == y
                         (x == 1)! == 3
                         x == y == y
                         x > y > z+3
                         x and y or z
                         not x and y or z
                         not x or y or z
                         ~x or y or z
                         x => y else z
                         x => y => z else w else q
                         x => y else z => w else q
                         not x => y else z
                         (1*((-24-(x^2))+(10*x)))/8
                         x/y/z
                         x*y*z
                         x+y+z
                         x-y-z
                         ((((((((((x+y))))))))))
                         ((((((((((x+y))))))))))[1]
                         -x!!!!!
                         ---x
                         (x-y) == (x+1)!
                         not x - y
                         not-+not - x
                         not x
                         - x
                         -1
                         not -x
                         - not x
                        """)

    grammar().validate()
