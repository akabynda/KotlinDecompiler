import math

_OPS = {
    '+', '-', '*', '/', '%', '++', '--', '+=', '-=', '*=', '/=', '%=',
    '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', '?', '?:', '?.',
    '::', '..', '..<', 'in', '!in', 'is', '!is', '=', '->', '@', 'as', 'as?'
}
_OPERAND_TYPES = {
    'identifier', 'this', 'super', 'integer_literal', 'float_literal',
    'string_literal', 'null_literal', 'boolean_literal', 'character_literal'
}


def _walk(n):
    if n.child_count == 0:
        yield n
    else:
        for c in n.children:
            yield from _walk(c)


def counts(root):
    N1 = N2 = 0
    ops = set()
    opr = set()
    for leaf in _walk(root):
        if leaf.text is None: continue
        txt = leaf.text.decode('utf8')
        if txt in _OPS:
            N1 += 1
            ops.add(txt)
        elif leaf.type in _OPERAND_TYPES:
            N2 += 1
            opr.add(txt)
    n1 = len(ops)
    n2 = len(opr)
    return N1, N2, n1, n2


def derived(root):
    N1, N2, n1, n2 = counts(root)
    N = N1 + N2
    n = n1 + n2
    V = N * math.log2(n) if n else 0
    D = (n1 / 2) * (N2 / n2) if n2 else 0
    E = D * V
    T = E / 18
    B = (E ** (2 / 3)) / 3000 if E else 0
    return {
        "Halstead Length": N,
        "Halstead Vocabulary": n,
        "Halstead Volume": V,
        "Halstead Difficulty": D,
        "Halstead Effort": E,
        "Halstead Time": T,
        "Halstead Bugs": B,
    }
