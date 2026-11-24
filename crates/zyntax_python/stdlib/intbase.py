# `int(text, base)`: a string read in a base from 2 to 36, or 0 to take
# it from a prefix.


def of_base(text, base):
    if not isinstance(text, str):
        raise TypeError("int() can't convert non-string with explicit base")
    if base != 0 and (base < 2 or base > 36):
        raise ValueError("int() base must be >= 2 and <= 36, or 0")
    s = text.strip()
    sign = 1
    if s.startswith("-"):
        sign = -1
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]
    if len(s) > 1 and s[0] == "0":
        p = s[1].lower()
        if p == "x" and (base == 16 or base == 0):
            base = 16
            s = s[2:]
        elif p == "o" and (base == 8 or base == 0):
            base = 8
            s = s[2:]
        elif p == "b" and (base == 2 or base == 0):
            base = 2
            s = s[2:]
    if base == 0:
        base = 10
    if s == "" or s.startswith("_") or s.endswith("_") or "__" in s:
        raise ValueError("invalid literal for int() with base " + str(base) + ": " + repr(text))
    value = 0
    for ch in s:
        if ch == "_":
            continue
        d = -1
        o = ord(ch)
        if o >= 48 and o <= 57:
            d = o - 48
        elif o >= 97 and o <= 122:
            d = o - 87
        elif o >= 65 and o <= 90:
            d = o - 55
        if d < 0 or d >= base:
            raise ValueError("invalid literal for int() with base " + str(base) + ": " + repr(text))
        value = value * base + d
    return sign * value
