# `json.loads` and `json.load`: JSON text read into dicts, lists, strings,
# numbers, booleans and None. `json.dumps` is the host's.


class JSONDecodeError(ValueError):
    pass


class _Reader:
    def __init__(self, text):
        self.text = text
        self.at = 0

    def fail(self, what):
        raise JSONDecodeError(what + ": line 1 column " + str(self.at + 1) + " (char " + str(self.at) + ")")

    def skip(self):
        while self.at < len(self.text) and self.text[self.at] in " \t\n\r":
            self.at += 1

    def value(self):
        self.skip()
        if self.at >= len(self.text):
            self.fail("Expecting value")
        c = self.text[self.at]
        if c == "{":
            return self.obj()
        if c == "[":
            return self.array()
        if c == '"':
            return self.string()
        if self.text.startswith("true", self.at):
            self.at += 4
            return True
        if self.text.startswith("false", self.at):
            self.at += 5
            return False
        if self.text.startswith("null", self.at):
            self.at += 4
            return None
        if self.text.startswith("NaN", self.at):
            self.at += 3
            return float("nan")
        if self.text.startswith("Infinity", self.at):
            self.at += 8
            return float("inf")
        if self.text.startswith("-Infinity", self.at):
            self.at += 9
            return float("-inf")
        return self.number()

    def number(self):
        start = self.at
        if self.at < len(self.text) and self.text[self.at] == "-":
            self.at += 1
        digits = 0
        while self.at < len(self.text) and self.text[self.at] in "0123456789":
            self.at += 1
            digits += 1
        if digits == 0:
            self.at = start
            self.fail("Expecting value")
        real = False
        if self.at < len(self.text) and self.text[self.at] == ".":
            real = True
            self.at += 1
            while self.at < len(self.text) and self.text[self.at] in "0123456789":
                self.at += 1
        if self.at < len(self.text) and self.text[self.at] in "eE":
            real = True
            self.at += 1
            if self.at < len(self.text) and self.text[self.at] in "+-":
                self.at += 1
            while self.at < len(self.text) and self.text[self.at] in "0123456789":
                self.at += 1
        piece = self.text[start:self.at]
        if real:
            return float(piece)
        return int(piece)

    def string(self):
        self.at += 1
        out = []
        while True:
            if self.at >= len(self.text):
                self.fail("Unterminated string starting at")
            c = self.text[self.at]
            if c == '"':
                self.at += 1
                return "".join(out)
            if c == "\\":
                self.at += 1
                if self.at >= len(self.text):
                    self.fail("Unterminated string starting at")
                e = self.text[self.at]
                if e == "n":
                    out.append("\n")
                elif e == "t":
                    out.append("\t")
                elif e == "r":
                    out.append("\r")
                elif e == "b":
                    out.append("\b")
                elif e == "f":
                    out.append("\f")
                elif e == "u":
                    code = 0
                    for digit in self.text[self.at + 1:self.at + 5].lower():
                        code = code * 16 + "0123456789abcdef".index(digit)
                    self.at += 4
                    out.append(chr(code))
                else:
                    out.append(e)
                self.at += 1
            else:
                out.append(c)
                self.at += 1

    def array(self):
        self.at += 1
        items = []
        self.skip()
        if self.at < len(self.text) and self.text[self.at] == "]":
            self.at += 1
            return items
        while True:
            items.append(self.value())
            self.skip()
            if self.at >= len(self.text):
                self.fail("Expecting ',' delimiter")
            if self.text[self.at] == ",":
                self.at += 1
            elif self.text[self.at] == "]":
                self.at += 1
                return items
            else:
                self.fail("Expecting ',' delimiter")

    def obj(self):
        self.at += 1
        out = {}
        self.skip()
        if self.at < len(self.text) and self.text[self.at] == "}":
            self.at += 1
            return out
        while True:
            self.skip()
            if self.at >= len(self.text) or self.text[self.at] != '"':
                self.fail("Expecting property name enclosed in double quotes")
            key = self.string()
            self.skip()
            if self.at >= len(self.text) or self.text[self.at] != ":":
                self.fail("Expecting ':' delimiter")
            self.at += 1
            out[key] = self.value()
            self.skip()
            if self.at >= len(self.text):
                self.fail("Expecting ',' delimiter")
            if self.text[self.at] == ",":
                self.at += 1
            elif self.text[self.at] == "}":
                self.at += 1
                return out
            else:
                self.fail("Expecting ',' delimiter")


def loads(text):
    reader = _Reader(text)
    result = reader.value()
    reader.skip()
    if reader.at < len(text):
        reader.fail("Extra data")
    return result


def load(f):
    return loads(f.read())
