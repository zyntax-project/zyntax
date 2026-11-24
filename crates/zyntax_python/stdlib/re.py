# Regular expressions: the patterns programs write most. A pattern is read
# into a tree, compiled to instructions, and run by a backtracking
# machine. Supported: literals, `.`, classes and `\d \w \s \b` and their
# negations, `^ $ \A \Z`, groups (numbered, `(?:)`, `(?P<name>)`),
# alternation, `* + ? {m,n}` greedy and lazy, `\1` and `(?P=name)`,
# lookahead `(?= )` and `(?! )`, and the flags I, M, S, X.

IGNORECASE = 2
I = 2
MULTILINE = 8
M = 8
DOTALL = 16
S = 16
VERBOSE = 64
X = 64
ASCII = 256
A = 256
UNICODE = 32
U = 32

CHAR = 1
ANY = 2
CLASS = 3
SPLIT = 4
JMP = 5
SAVE = 6
MATCH = 7
BOL = 8
EOL = 9
WORDB = 10
NWORDB = 11
BACKREF = 12
LOOK = 13
LOOKEND = 14
TEXTSTART = 15
TEXTEND = 16


class error(Exception):
    pass


def _is_word(c):
    if c >= 48 and c <= 57:
        return True
    if c >= 65 and c <= 90:
        return True
    if c >= 97 and c <= 122:
        return True
    if c == 95:
        return True
    if c > 127:
        return chr(c).isalnum()
    return False


def _is_space(c):
    if c >= 9 and c <= 13:
        return True
    if c == 32:
        return True
    if c > 127:
        return chr(c).isspace()
    return False


def _is_digit(c):
    if c >= 48 and c <= 57:
        return True
    if c > 127:
        return chr(c).isdigit()
    return False


class _Parser:
    def __init__(self, pattern, flags):
        self.pat = pattern
        self.i = 0
        self.flags = flags
        self.groups = 0
        self.names = {}
        self.backrefs = False
        self.looks = False

    def fail(self, what):
        raise error(what + " at position " + str(self.i))

    def more(self):
        return self.i < len(self.pat)

    def peek(self):
        return self.pat[self.i]

    def parse(self):
        node = self.alt()
        if self.more():
            self.fail("unbalanced parenthesis")
        return node

    def alt(self):
        branches = [self.seq()]
        while self.more() and self.peek() == "|":
            self.i += 1
            branches.append(self.seq())
        if len(branches) == 1:
            return branches[0]
        return ["alt", branches]

    def seq(self):
        items = []
        while self.more():
            c = self.peek()
            if c == "|" or c == ")":
                break
            if (self.flags & VERBOSE) != 0:
                if c in " \t\n\r":
                    self.i += 1
                    continue
                if c == "#":
                    while self.more() and self.peek() != "\n":
                        self.i += 1
                    continue
            atom = self.atom()
            if atom is None:
                continue
            items.append(self.quantified(atom))
        return ["seq", items]

    def quantified(self, atom):
        while self.more():
            c = self.peek()
            lo = 0
            hi = -1
            if c == "*":
                self.i += 1
            elif c == "+":
                lo = 1
                self.i += 1
            elif c == "?":
                hi = 1
                self.i += 1
            elif c == "{":
                bounds = self.braces()
                if bounds is None:
                    return atom
                lo = bounds[0]
                hi = bounds[1]
            else:
                return atom
            greedy = True
            if self.more() and self.peek() == "?":
                greedy = False
                self.i += 1
            elif self.more() and self.peek() == "+":
                self.i += 1
            if atom[0] == "bol" or atom[0] == "eol":
                self.fail("nothing to repeat")
            atom = ["rep", atom, lo, hi, greedy]
        return atom

    def braces(self):
        # `{m}`, `{m,}`, `{,n}`, `{m,n}`; anything else is a literal brace.
        j = self.i + 1
        text = ""
        while j < len(self.pat) and self.pat[j] != "}":
            text += self.pat[j]
            j += 1
        if j >= len(self.pat):
            return None
        parts = text.split(",")
        if len(parts) == 1:
            if not parts[0].isdigit():
                return None
            self.i = j + 1
            n = int(parts[0])
            return [n, n]
        if len(parts) != 2:
            return None
        lo = 0
        hi = -1
        if parts[0] != "":
            if not parts[0].isdigit():
                return None
            lo = int(parts[0])
        if parts[1] != "":
            if not parts[1].isdigit():
                return None
            hi = int(parts[1])
        if parts[0] == "" and parts[1] == "":
            return None
        self.i = j + 1
        return [lo, hi]

    def atom(self):
        c = self.peek()
        self.i += 1
        if c == "(":
            return self.parse_group()
        if c == "[":
            return self.charclass()
        if c == ".":
            return ["any"]
        if c == "^":
            return ["bol"]
        if c == "$":
            return ["eol"]
        if c == "\\":
            return self.escape()
        if c == "*" or c == "+" or c == "?":
            self.i -= 1
            self.fail("nothing to repeat")
        return ["char", ord(c)]

    def parse_group(self):
        if self.more() and self.peek() == "?":
            self.i += 1
            if not self.more():
                self.fail("unexpected end of pattern")
            k = self.peek()
            self.i += 1
            if k == ":":
                node = self.alt()
                self.close()
                return node
            if k == "P":
                if not self.more():
                    self.fail("unexpected end of pattern")
                k2 = self.peek()
                self.i += 1
                if k2 == "<":
                    name = self.upto(">")
                    self.groups += 1
                    index = self.groups
                    self.names[name] = index
                    node = self.alt()
                    self.close()
                    return ["group", index, node]
                if k2 == "=":
                    name = self.upto(")")
                    if name not in self.names:
                        self.fail("unknown group name")
                    self.backrefs = True
                    return ["backref", self.names[name]]
                self.fail("unknown extension")
            if k == "=" or k == "!":
                self.looks = True
                node = self.alt()
                self.close()
                return ["look", node, k == "!"]
            if k == "#":
                self.upto(")")
                return None
            # Inline flags, `(?i)`.
            self.i -= 1
            while self.more() and self.peek() in "imsxaLu":
                f = self.peek()
                if f == "i":
                    self.flags |= IGNORECASE
                elif f == "m":
                    self.flags |= MULTILINE
                elif f == "s":
                    self.flags |= DOTALL
                elif f == "x":
                    self.flags |= VERBOSE
                self.i += 1
            if self.more() and self.peek() == ")":
                self.i += 1
                return None
            self.fail("unknown extension")
        self.groups += 1
        index = self.groups
        node = self.alt()
        self.close()
        return ["group", index, node]

    def upto(self, end):
        out = ""
        while self.more() and self.peek() != end:
            out += self.peek()
            self.i += 1
        if not self.more():
            self.fail("missing " + end)
        self.i += 1
        return out

    def close(self):
        if not self.more() or self.peek() != ")":
            self.fail("missing ), unterminated subpattern")
        self.i += 1

    def escape(self):
        if not self.more():
            self.fail("bad escape (end of pattern)")
        c = self.peek()
        self.i += 1
        if c == "d":
            return ["class", False, [["cat", "d", False]]]
        if c == "D":
            return ["class", True, [["cat", "d", False]]]
        if c == "w":
            return ["class", False, [["cat", "w", False]]]
        if c == "W":
            return ["class", True, [["cat", "w", False]]]
        if c == "s":
            return ["class", False, [["cat", "s", False]]]
        if c == "S":
            return ["class", True, [["cat", "s", False]]]
        if c == "b":
            return ["wordb", False]
        if c == "B":
            return ["wordb", True]
        if c == "A":
            return ["textstart"]
        if c == "Z" or c == "z":
            return ["textend"]
        if c == "n":
            return ["char", 10]
        if c == "t":
            return ["char", 9]
        if c == "r":
            return ["char", 13]
        if c == "f":
            return ["char", 12]
        if c == "v":
            return ["char", 11]
        if c == "0":
            return ["char", 0]
        if c.isdigit():
            n = int(c)
            while self.more() and self.peek().isdigit() and n * 10 + int(self.peek()) <= self.groups:
                n = n * 10 + int(self.peek())
                self.i += 1
            self.backrefs = True
            return ["backref", n]
        if c == "x":
            code = 0
            for digit in self.pat[self.i:self.i + 2].lower():
                code = code * 16 + "0123456789abcdef".index(digit)
            self.i += 2
            return ["char", code]
        if c.isalpha():
            self.i -= 1
            self.fail("bad escape \\" + c)
        return ["char", ord(c)]

    def charclass(self):
        negated = False
        if self.more() and self.peek() == "^":
            negated = True
            self.i += 1
        items = []
        first = True
        while True:
            if not self.more():
                self.fail("unterminated character set")
            c = self.peek()
            if c == "]" and not first:
                self.i += 1
                break
            first = False
            self.i += 1
            lo = ord(c)
            if c == "\\":
                if not self.more():
                    self.fail("bad escape (end of pattern)")
                e = self.peek()
                self.i += 1
                if e == "d" or e == "w" or e == "s":
                    items.append(["cat", e, False])
                    continue
                if e == "D" or e == "W" or e == "S":
                    items.append(["cat", e.lower(), True])
                    continue
                if e == "n":
                    lo = 10
                elif e == "t":
                    lo = 9
                elif e == "r":
                    lo = 13
                elif e == "b":
                    lo = 8
                else:
                    lo = ord(e)
            hi = lo
            if self.i + 1 < len(self.pat) and self.peek() == "-" and self.pat[self.i + 1] != "]":
                self.i += 1
                h = self.peek()
                self.i += 1
                hi = ord(h)
                if h == "\\":
                    e = self.peek()
                    self.i += 1
                    hi = ord(e)
                    if e == "n":
                        hi = 10
                    elif e == "t":
                        hi = 9
                if hi < lo:
                    self.fail("bad character range")
            items.append(["range", lo, hi])
        return ["class", negated, items]


def _class_has(items, c):
    for item in items:
        if item[0] == "range":
            if c >= item[1] and c <= item[2]:
                return True
        else:
            kind = item[1]
            if kind == "d":
                hit = _is_digit(c)
            elif kind == "w":
                hit = _is_word(c)
            else:
                hit = _is_space(c)
            if hit != item[2]:
                return True
    return False


class _Program:
    def __init__(self):
        self.ops = []
        self.x = []
        self.y = []
        self.classes = []

    def add(self, op, x, y):
        self.ops.append(op)
        self.x.append(x)
        self.y.append(y)
        return len(self.ops) - 1

    def emit(self, node, icase):
        kind = node[0]
        if kind == "seq":
            for item in node[1]:
                self.emit(item, icase)
        elif kind == "char":
            code = node[1]
            if icase:
                code = ord(chr(code).lower())
            self.add(CHAR, code, 0)
        elif kind == "any":
            self.add(ANY, 0, 0)
        elif kind == "class":
            self.classes.append([node[1], node[2]])
            self.add(CLASS, len(self.classes) - 1, 0)
        elif kind == "bol":
            self.add(BOL, 0, 0)
        elif kind == "eol":
            self.add(EOL, 0, 0)
        elif kind == "textstart":
            self.add(TEXTSTART, 0, 0)
        elif kind == "textend":
            self.add(TEXTEND, 0, 0)
        elif kind == "wordb":
            if node[1]:
                self.add(NWORDB, 0, 0)
            else:
                self.add(WORDB, 0, 0)
        elif kind == "backref":
            self.add(BACKREF, node[1], 0)
        elif kind == "group":
            self.add(SAVE, node[1] * 2, 0)
            self.emit(node[2], icase)
            self.add(SAVE, node[1] * 2 + 1, 0)
        elif kind == "look":
            at = self.add(LOOK, 0, 0)
            if node[2]:
                self.y[at] = 1
            self.emit(node[1], icase)
            self.add(LOOKEND, 0, 0)
            self.x[at] = len(self.ops)
        elif kind == "alt":
            branches = node[1]
            jumps = []
            i = 0
            while i < len(branches):
                if i < len(branches) - 1:
                    split = self.add(SPLIT, 0, 0)
                    self.x[split] = len(self.ops)
                    self.emit(branches[i], icase)
                    jumps.append(self.add(JMP, 0, 0))
                    self.y[split] = len(self.ops)
                else:
                    self.emit(branches[i], icase)
                i += 1
            for j in jumps:
                self.x[j] = len(self.ops)
        elif kind == "rep":
            self.repeat(node, icase)

    def repeat(self, node, icase):
        body = node[1]
        lo = node[2]
        hi = node[3]
        greedy = node[4]
        i = 0
        while i < lo:
            self.emit(body, icase)
            i += 1
        if hi == -1:
            # body* after the required copies.
            split = self.add(SPLIT, 0, 0)
            start = len(self.ops)
            self.emit(body, icase)
            self.add(JMP, split, 0)
            out = len(self.ops)
            if greedy:
                self.x[split] = start
                self.y[split] = out
            else:
                self.x[split] = out
                self.y[split] = start
        else:
            splits = []
            i = lo
            while i < hi:
                split = self.add(SPLIT, 0, 0)
                splits.append(split)
                start = len(self.ops)
                if greedy:
                    self.x[split] = start
                else:
                    self.y[split] = start
                self.emit(body, icase)
                i += 1
            out = len(self.ops)
            for split in splits:
                if greedy:
                    self.y[split] = out
                else:
                    self.x[split] = out


class Match:
    def __init__(self, pattern, string, caps, pos, endpos):
        self.re = pattern
        self.string = string
        self.caps = caps
        self.pos = pos
        self.endpos = endpos

    def _index(self, group):
        if isinstance(group, str):
            if group not in self.re.groupindex:
                raise IndexError("no such group")
            return self.re.groupindex[group]
        if group < 0 or group > self.re.groups:
            raise IndexError("no such group")
        return group

    def group(self, *which):
        if len(which) == 0:
            return self.string[self.caps[0]:self.caps[1]]
        if len(which) == 1:
            return self._one(which[0])
        out = []
        for w in which:
            out.append(self._one(w))
        return tuple(out)

    def _one(self, w):
        i = self._index(w)
        if self.caps[2 * i] < 0:
            return None
        return self.string[self.caps[2 * i]:self.caps[2 * i + 1]]

    def __getitem__(self, w):
        return self._one(w)

    def groups(self, default=None):
        out = []
        i = 1
        while i <= self.re.groups:
            value = self._one(i)
            if value is None:
                value = default
            out.append(value)
            i += 1
        return tuple(out)

    def groupdict(self, default=None):
        out = {}
        for name in self.re.groupindex:
            value = self._one(name)
            if value is None:
                value = default
            out[name] = value
        return out

    def start(self, group=0):
        return self.caps[2 * self._index(group)]

    def end(self, group=0):
        return self.caps[2 * self._index(group) + 1]

    def span(self, group=0):
        i = self._index(group)
        return (self.caps[2 * i], self.caps[2 * i + 1])

    def expand(self, template):
        return _expand(self, template)

    def __repr__(self):
        return "<re.Match object; span=" + str(self.span()) + ", match=" + repr(self.group()) + ">"


class _ListIter:
    def __init__(self, items):
        self.items = items
        self.at = 0

    def __next__(self):
        if self.at >= len(self.items):
            raise StopIteration
        item = self.items[self.at]
        self.at += 1
        return item


def _expand(m, template):
    out = []
    i = 0
    while i < len(template):
        c = template[i]
        if c == "\\" and i + 1 < len(template):
            n = template[i + 1]
            if n.isdigit():
                value = m.group(int(n))
                if value is not None:
                    out.append(value)
                i += 2
            elif n == "g" and i + 2 < len(template) and template[i + 2] == "<":
                end = template.find(">", i + 3)
                name = template[i + 3:end]
                key = name
                if name.isdigit():
                    key = int(name)
                value = m.group(key)
                if value is not None:
                    out.append(value)
                i = end + 1
            elif n == "n":
                out.append("\n")
                i += 2
            elif n == "t":
                out.append("\t")
                i += 2
            elif n == "\\":
                out.append("\\")
                i += 2
            else:
                out.append(c)
                out.append(n)
                i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


class Pattern:
    def __init__(self, pattern, flags):
        parser = _Parser(pattern, flags)
        tree = parser.parse()
        self.pattern = pattern
        self.flags = parser.flags
        self.groups = parser.groups
        self.groupindex = parser.names
        self.exact = not parser.backrefs and not parser.looks
        self.icase = (parser.flags & IGNORECASE) != 0
        self.multiline = (parser.flags & MULTILINE) != 0
        self.dotall = (parser.flags & DOTALL) != 0
        self.prog = _Program()
        self.prog.add(SAVE, 0, 0)
        self.prog.emit(tree, self.icase)
        self.prog.add(SAVE, 1, 0)
        self.prog.add(MATCH, 0, 0)

    def _run(self, text, codes, pc, pos, caps, seen):
        prog = self.prog
        ops = prog.ops
        xs = prog.x
        ys = prog.y
        n = len(text)
        stack = []
        steps = 0
        while True:
            op = ops[pc]
            ok = True
            if op == CHAR:
                if pos < n and codes[pos] == xs[pc]:
                    pos += 1
                    pc += 1
                else:
                    ok = False
            elif op == ANY:
                if pos < n and (self.dotall or codes[pos] != 10):
                    pos += 1
                    pc += 1
                else:
                    ok = False
            elif op == CLASS:
                if pos < n:
                    entry = prog.classes[xs[pc]]
                    c = codes[pos]
                    hit = _class_has(entry[1], c)
                    if not hit and self.icase:
                        low = ord(chr(c).lower())
                        up = ord(chr(c).upper())
                        hit = _class_has(entry[1], low) or _class_has(entry[1], up)
                    if hit != entry[0]:
                        pos += 1
                        pc += 1
                    else:
                        ok = False
                else:
                    ok = False
            elif op == SPLIT:
                key = pc * (n + 1) + pos
                if self.exact and key in seen:
                    ok = False
                else:
                    if self.exact:
                        seen.add(key)
                    stack.append((ys[pc], pos, caps[:]))
                    pc = xs[pc]
            elif op == JMP:
                pc = xs[pc]
            elif op == SAVE:
                caps[xs[pc]] = pos
                pc += 1
            elif op == MATCH:
                return pos
            elif op == BOL:
                if pos == 0 or (self.multiline and codes[pos - 1] == 10):
                    pc += 1
                else:
                    ok = False
            elif op == EOL:
                if pos == n or (self.multiline and codes[pos] == 10) or (pos == n - 1 and codes[pos] == 10):
                    pc += 1
                else:
                    ok = False
            elif op == TEXTSTART:
                if pos == 0:
                    pc += 1
                else:
                    ok = False
            elif op == TEXTEND:
                if pos == n:
                    pc += 1
                else:
                    ok = False
            elif op == WORDB or op == NWORDB:
                before = pos > 0 and _is_word(codes[pos - 1])
                after = pos < n and _is_word(codes[pos])
                if (before != after) == (op == WORDB):
                    pc += 1
                else:
                    ok = False
            elif op == BACKREF:
                a = caps[2 * xs[pc]]
                b = caps[2 * xs[pc] + 1]
                if a < 0:
                    ok = False
                else:
                    size = b - a
                    if pos + size <= n and codes[pos:pos + size] == codes[a:b]:
                        pos += size
                        pc += 1
                    else:
                        ok = False
            elif op == LOOK:
                inner = caps[:]
                found = self._run(text, codes, pc + 1, pos, inner, set())
                if (found >= 0) != (ys[pc] == 1):
                    pc = xs[pc]
                else:
                    ok = False
            elif op == LOOKEND:
                return pos
            if not ok:
                if len(stack) == 0:
                    return -1
                entry = stack.pop()
                pc = entry[0]
                pos = entry[1]
                caps[:] = entry[2]
                steps += 1
                if steps > 5000000:
                    raise error("pattern too complex")

    def _codes(self, string):
        codes = []
        for ch in string:
            if self.icase:
                codes.append(ord(ch.lower()))
            else:
                codes.append(ord(ch))
        return codes

    def _try(self, string, codes, start, anchored_end, seen):
        caps = [-1] * (2 * (self.groups + 1))
        end = self._run(string, codes, 0, start, caps, seen)
        if end < 0:
            return None
        if anchored_end and end != len(string):
            return None
        return caps

    def match(self, string, pos=0, endpos=None):
        if endpos is None:
            endpos = len(string)
        text = string[:endpos]
        caps = self._try(text, self._codes(text), pos, False, set())
        if caps is None:
            return None
        return Match(self, string, caps, pos, endpos)

    def fullmatch(self, string, pos=0, endpos=None):
        if endpos is None:
            endpos = len(string)
        text = string[:endpos]
        codes = self._codes(text)
        # Every way the pattern can match from pos, until one ends at the end.
        caps = self._try_full(text, codes, pos)
        if caps is None:
            return None
        return Match(self, string, caps, pos, endpos)

    def _try_full(self, text, codes, pos):
        # The pattern followed by an end-of-text test.
        wrapped = Pattern("(?:" + self.pattern + ")\\Z", self.flags)
        caps = wrapped._try(text, codes, pos, False, set())
        if caps is None:
            return None
        return caps[:2 * (self.groups + 1)]

    def search(self, string, pos=0, endpos=None):
        if endpos is None:
            endpos = len(string)
        text = string[:endpos]
        codes = self._codes(text)
        seen = set()
        start = pos
        while start <= len(text):
            caps = self._try(text, codes, start, False, seen)
            if caps is not None:
                return Match(self, string, caps, pos, endpos)
            start += 1
        return None

    def finditer(self, string, pos=0, endpos=None):
        return _ListIter(self._all(string, pos, endpos))

    def _all(self, string, pos, endpos):
        if endpos is None:
            endpos = len(string)
        text = string[:endpos]
        codes = self._codes(text)
        out = []
        start = pos
        while start <= len(text):
            caps = self._try(text, codes, start, False, set())
            if caps is None:
                start += 1
                continue
            out.append(Match(self, string, caps, pos, endpos))
            if caps[1] == caps[0]:
                start = caps[1] + 1
            else:
                start = caps[1]
        return out

    def findall(self, string, pos=0, endpos=None):
        out = []
        for m in self._all(string, pos, endpos):
            if self.groups == 0:
                out.append(m.group())
            elif self.groups == 1:
                value = m.group(1)
                if value is None:
                    value = ""
                out.append(value)
            else:
                row = []
                i = 1
                while i <= self.groups:
                    value = m.group(i)
                    if value is None:
                        value = ""
                    row.append(value)
                    i += 1
                out.append(tuple(row))
        return out

    def subn(self, repl, string, count=0):
        out = []
        last = 0
        n = 0
        for m in self._all(string, 0, None):
            if count > 0 and n >= count:
                break
            out.append(string[last:m.start()])
            if isinstance(repl, str):
                out.append(_expand(m, repl))
            else:
                out.append(repl(m))
            last = m.end()
            n += 1
        out.append(string[last:])
        return "".join(out), n

    def sub(self, repl, string, count=0):
        return self.subn(repl, string, count)[0]

    def split(self, string, maxsplit=0):
        out = []
        last = 0
        n = 0
        for m in self._all(string, 0, None):
            if maxsplit > 0 and n >= maxsplit:
                break
            if m.end() == m.start() and m.start() == 0:
                continue
            out.append(string[last:m.start()])
            i = 1
            while i <= self.groups:
                out.append(m.group(i))
                i += 1
            last = m.end()
            n += 1
        out.append(string[last:])
        return out

    def __repr__(self):
        return "re.compile(" + repr(self.pattern) + ")"


def compile(pattern, flags=0):
    if isinstance(pattern, Pattern):
        return pattern
    return Pattern(pattern, flags)


def match(pattern, string, flags=0):
    return compile(pattern, flags).match(string)


def fullmatch(pattern, string, flags=0):
    return compile(pattern, flags).fullmatch(string)


def search(pattern, string, flags=0):
    return compile(pattern, flags).search(string)


def findall(pattern, string, flags=0):
    return compile(pattern, flags).findall(string)


def finditer(pattern, string, flags=0):
    return compile(pattern, flags).finditer(string)


def sub(pattern, repl, string, count=0, flags=0):
    return compile(pattern, flags).sub(repl, string, count)


def subn(pattern, repl, string, count=0, flags=0):
    return compile(pattern, flags).subn(repl, string, count)


def split(pattern, string, maxsplit=0, flags=0):
    return compile(pattern, flags).split(string, maxsplit)


def escape(text):
    out = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("\\" + ch)
    return "".join(out)
