# Arbitrary-precision int, for a program that opts in. An int that fits
# 64 bits is the machine int; one that does not is a BigInt of a sign and
# little-endian 60-bit limbs. The operators that can overflow call the
# functions at the end of this file, which check and promote; a result that
# fits 64 bits comes back as the machine int, so a BigInt is never equal to
# one.

import __bignative

LIMB = 60
MASK = 1152921504606846975
MIN = -9223372036854775807 - 1


class BigInt:
    def __init__(self, sign: int, mag: list[int]):
        self.sign = sign
        self.mag = mag

    def __add__(self, other):
        if _isint(other):
            return _addsub(self, other, False)
        if isinstance(other, float):
            return float(self) + other
        raise TypeError("unsupported operand type(s) for +: 'int' and '" + type(other).__name__ + "'")

    def __radd__(self, other):
        if _isint(other):
            return _addsub(other, self, False)
        if isinstance(other, float):
            return other + float(self)
        raise TypeError("unsupported operand type(s) for +: '" + type(other).__name__ + "' and 'int'")

    def __sub__(self, other):
        if _isint(other):
            return _addsub(self, other, True)
        if isinstance(other, float):
            return float(self) - other
        raise TypeError("unsupported operand type(s) for -: 'int' and '" + type(other).__name__ + "'")

    def __rsub__(self, other):
        if _isint(other):
            return _addsub(other, self, True)
        if isinstance(other, float):
            return other - float(self)
        raise TypeError("unsupported operand type(s) for -: '" + type(other).__name__ + "' and 'int'")

    def __mul__(self, other):
        if _isint(other):
            return _mulvals(self, other)
        if isinstance(other, float):
            return float(self) * other
        raise TypeError("unsupported operand type(s) for *: 'int' and '" + type(other).__name__ + "'")

    def __rmul__(self, other):
        if _isint(other):
            return _mulvals(other, self)
        if isinstance(other, float):
            return other * float(self)
        raise TypeError("unsupported operand type(s) for *: '" + type(other).__name__ + "' and 'int'")

    def __floordiv__(self, other):
        if _isint(other):
            return _divmodvals(self, other)[0]
        if isinstance(other, float):
            return float(self) // other
        raise TypeError("unsupported operand type(s) for //: 'int' and '" + type(other).__name__ + "'")

    def __rfloordiv__(self, other):
        if _isint(other):
            return _divmodvals(other, self)[0]
        if isinstance(other, float):
            return other // float(self)
        raise TypeError("unsupported operand type(s) for //: '" + type(other).__name__ + "' and 'int'")

    def __mod__(self, other):
        if _isint(other):
            return _divmodvals(self, other)[1]
        if isinstance(other, float):
            return float(self) % other
        raise TypeError("unsupported operand type(s) for %: 'int' and '" + type(other).__name__ + "'")

    def __rmod__(self, other):
        if _isint(other):
            return _divmodvals(other, self)[1]
        if isinstance(other, float):
            return other % float(self)
        raise TypeError("unsupported operand type(s) for %: '" + type(other).__name__ + "' and 'int'")

    def __divmod__(self, other):
        return _divmodvals(self, other)

    def __rdivmod__(self, other):
        return _divmodvals(other, self)

    def __truediv__(self, other):
        if _isint(other):
            return _truediv(self, other)
        if isinstance(other, float):
            return float(self) / other
        raise TypeError("unsupported operand type(s) for /: 'int' and '" + type(other).__name__ + "'")

    def __rtruediv__(self, other):
        if _isint(other):
            return _truediv(other, self)
        if isinstance(other, float):
            return other / float(self)
        raise TypeError("unsupported operand type(s) for /: '" + type(other).__name__ + "' and 'int'")

    def __pow__(self, other):
        return power(self, other)

    def __rpow__(self, other):
        return power(other, self)

    def __lshift__(self, other):
        return lshift(self, other)

    def __rlshift__(self, other):
        return lshift(other, self)

    def __rshift__(self, other):
        return _rshift(self, other)

    def __rrshift__(self, other):
        return _rshift(other, self)

    def __and__(self, other):
        return _bitop(0, self, other)

    def __rand__(self, other):
        return _bitop(0, other, self)

    def __or__(self, other):
        return _bitop(1, self, other)

    def __ror__(self, other):
        return _bitop(1, other, self)

    def __xor__(self, other):
        return _bitop(2, self, other)

    def __rxor__(self, other):
        return _bitop(2, other, self)

    def __invert__(self):
        return _addsub(_negate(self), 1, True)

    def __neg__(self):
        return _negate(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return BigInt(1, self.mag)

    def __lt__(self, other):
        return _compare(self, other) < 0

    def __le__(self, other):
        return _compare(self, other) <= 0

    def __gt__(self, other):
        return _compare(self, other) > 0

    def __ge__(self, other):
        return _compare(self, other) >= 0

    def __eq__(self, other):
        if isinstance(other, BigInt):
            return self.sign == other.sign and _cmp_mag(self.mag, other.mag) == 0
        if isinstance(other, float):
            return float(self) == other
        return False

    def __ne__(self, other):
        if isinstance(other, BigInt):
            return not (self.sign == other.sign and _cmp_mag(self.mag, other.mag) == 0)
        if isinstance(other, float):
            return float(self) != other
        return True

    def __hash__(self):
        # The hash of an int: its value modulo 2 ** 61 - 1.
        p = 2305843009213693951
        h = 0
        i = len(self.mag) - 1
        while i >= 0:
            # h * 2 ** 60 modulo 2 ** 61 - 1, as the bit moved across.
            h = (h >> 1) + ((h & 1) << 60) + self.mag[i]
            while h > p:
                h = (h & p) + (h >> 61)
            if h == p:
                h = 0
            i -= 1
        if self.sign < 0:
            h = -h
            if h == -1:
                h = -2
        return h

    def __bool__(self):
        return True

    def __int__(self):
        return self

    def __index__(self):
        return self

    def __float__(self):
        return _tofloat(self)

    def __str__(self):
        if self.sign < 0:
            return "-" + _mag_str(self.mag)
        return _mag_str(self.mag)

    def __repr__(self):
        return self.__str__()

    def __format__(self, spec):
        return _format(self, spec)

    def bit_length(self):
        return (len(self.mag) - 1) * LIMB + _limb_bits(self.mag[len(self.mag) - 1])


def _isint(x: object) -> bool:
    return isinstance(x, int) or isinstance(x, BigInt)


def _trim(m: list[int]) -> list[int]:
    while len(m) > 0 and m[len(m) - 1] == 0:
        m.pop()
    return m


def _limbs(x: int) -> list[int]:
    m: list[int] = []
    while x > 0:
        m.append(x & MASK)
        x = x >> LIMB
    return m


def _of_int(x: int) -> tuple[int, list[int]]:
    if x == 0:
        return 1, []
    if x < 0:
        if x == MIN:
            return -1, [0, 8]
        return -1, _limbs(-x)
    return 1, _limbs(x)


def _parts(x: object) -> tuple[int, list[int]]:
    if isinstance(x, BigInt):
        return x.sign, x.mag
    return _of_int(x)


def _make(sign: int, m: list[int]) -> object:
    _trim(m)
    n = len(m)
    if n == 0:
        return 0
    if n == 1 or (n == 2 and m[1] < 8):
        v = 0
        i = n - 1
        while i >= 0:
            v = (v << LIMB) | m[i]
            i -= 1
        if sign < 0:
            return -v
        return v
    if sign < 0 and n == 2 and m[1] == 8 and m[0] == 0:
        return MIN
    return BigInt(sign, m)


def _limb_bits(x: int) -> int:
    n = 0
    while x > 0:
        n += 1
        x = x >> 1
    return n


def _cmp_mag(a: list[int], b: list[int]) -> int:
    return __bignative.cmp(a, b)


def _add_mag(a: list[int], b: list[int]) -> list[int]:
    n = len(a)
    if len(b) > n:
        n = len(b)
    out = __bignative.alloc(n + 1)
    __bignative.add(a, b, out)
    return _trim(out)


def _sub_mag(a: list[int], b: list[int]) -> list[int]:
    # a >= b
    out = __bignative.alloc(len(a))
    __bignative.sub(a, b, out)
    return _trim(out)


def _mul_mag(a: list[int], b: list[int]) -> list[int]:
    n = len(a)
    m = len(b)
    if n == 0 or m == 0:
        return []
    out = __bignative.alloc(n + m)
    __bignative.mul(a, b, out)
    return _trim(out)


def _divmod_small(a: list[int], d: int) -> tuple[list[int], int]:
    q = __bignative.alloc(len(a))
    r = __bignative.divmod_small(a, d, q)
    return _trim(q), r


def _shl_mag(a: list[int], n: int) -> list[int]:
    if len(a) == 0:
        return []
    out = __bignative.alloc(len(a) + n // LIMB + 1)
    __bignative.shl(a, n, out)
    return _trim(out)


def _shr_mag(a: list[int], n: int) -> list[int]:
    out = __bignative.alloc(len(a))
    __bignative.shr(a, n, out)
    return _trim(out)


def _divmod_mag(a: list[int], b: list[int]) -> tuple[list[int], list[int]]:
    # b is not empty.
    q = __bignative.alloc(len(a) + 1)
    r = __bignative.alloc(len(b))
    __bignative.divmod(a, b, q, r)
    return _trim(q), _trim(r)


def _addsub(a: object, b: object, negate: bool) -> object:
    sa, ma = _parts(a)
    sb, mb = _parts(b)
    if negate:
        sb = -sb
    if sa == sb:
        return _make(sa, _add_mag(ma, mb))
    c = _cmp_mag(ma, mb)
    if c == 0:
        return 0
    if c > 0:
        return _make(sa, _sub_mag(ma, mb))
    return _make(sb, _sub_mag(mb, ma))


def _negate(a: object) -> object:
    sa, ma = _parts(a)
    return _make(-sa, list(ma))


def _mulvals(a: object, b: object) -> object:
    sa, ma = _parts(a)
    sb, mb = _parts(b)
    return _make(sa * sb, _mul_mag(ma, mb))


def _divmodvals(a: object, b: object) -> tuple[object, object]:
    if not _isint(a) or not _isint(b):
        raise TypeError("unsupported operand type(s) for divmod()")
    sa, ma = _parts(a)
    sb, mb = _parts(b)
    if len(mb) == 0:
        raise ZeroDivisionError("division by zero")
    q, r = _divmod_mag(ma, mb)
    if sa != sb and len(r) > 0:
        q = _add_mag(q, [1])
        r = _sub_mag(mb, r)
    return _make(sa * sb, q), _make(sb, r)


def _compare(a: object, b: object) -> int:
    if isinstance(b, float):
        x = _tofloat(a)
        if x < b:
            return -1
        if x > b:
            return 1
        return 0
    if not _isint(b):
        raise TypeError("'<' not supported between instances of 'int' and '" + type(b).__name__ + "'")
    sa, ma = _parts(a)
    sb, mb = _parts(b)
    za = len(ma) == 0
    zb = len(mb) == 0
    if za and zb:
        return 0
    if za:
        return -sb
    if zb:
        return sa
    if sa != sb:
        if sa > sb:
            return 1
        return -1
    return _cmp_mag(ma, mb) * sa


def _bits_of(m: list[int]) -> int:
    if len(m) == 0:
        return 0
    return (len(m) - 1) * LIMB + _limb_bits(m[len(m) - 1])


def _tofloat(a: object) -> float:
    sa, ma = _parts(a)
    n = _bits_of(ma)
    if n <= 62:
        v = 0
        i = len(ma) - 1
        while i >= 0:
            v = (v << LIMB) | ma[i]
            i -= 1
        f = float(v)
    else:
        shift = n - 56
        top = _shr_mag(ma, shift)
        # The bits shifted out decide a tie, so keep one if any is set.
        sticky = _cmp_mag(_shl_mag(top, shift), ma) != 0
        v = 0
        i = len(top) - 1
        while i >= 0:
            v = (v << LIMB) | top[i]
            i -= 1
        if sticky:
            v = v | 1
        if shift > 1100:
            raise OverflowError("int too large to convert to float")
        half = shift // 2
        f = float(v) * (2.0 ** half) * (2.0 ** (shift - half))
        if f == float("inf"):
            raise OverflowError("int too large to convert to float")
    if sa < 0:
        return -f
    return f


def _from_float(f: float) -> object:
    if f != f:
        raise ValueError("cannot convert float NaN to integer")
    if f == float("inf") or f == float("-inf"):
        raise OverflowError("cannot convert float infinity to integer")
    neg = f < 0
    if neg:
        f = -f
    if f < 4611686018427387904.0:
        v = int(f)
        if neg:
            return -v
        return v
    e = 0
    while f >= 4611686018427387904.0:
        f = f / 2.0
        e += 1
    v = int(f)
    sv, mv = _of_int(v)
    r = _make(1, _shl_mag(mv, e))
    if neg:
        return _negate(r)
    return r


def _truediv(a: object, b: object) -> float:
    sa, ma = _parts(a)
    sb, mb = _parts(b)
    if len(mb) == 0:
        raise ZeroDivisionError("division by zero")
    if len(ma) == 0:
        return 0.0
    # Scale so the integer quotient has 56 bits or more, then round once.
    shift = _bits_of(mb) - _bits_of(ma) + 57
    if shift > 0:
        num = _shl_mag(ma, shift)
        den = mb
    else:
        num = ma
        den = _shl_mag(mb, -shift)
    q, r = _divmod_mag(num, den)
    v = 0
    i = len(q) - 1
    while i >= 0:
        v = (v << LIMB) | q[i]
        i -= 1
    if len(r) > 0:
        v = v | 1
    f = float(v)
    e = -shift
    half = e // 2
    f = f * (2.0 ** half) * (2.0 ** (e - half))
    if sa * sb < 0:
        return -f
    return f


def _mag_str(m: list[int]) -> str:
    if len(m) == 0:
        return "0"
    parts: list[int] = []
    cur = m
    while len(cur) > 0:
        cur, r = _divmod_small(cur, 1000000000000000000)
        parts.append(r)
    out = [str(parts[len(parts) - 1])]
    i = len(parts) - 2
    while i >= 0:
        s = str(parts[i])
        out.append("0" * (18 - len(s)) + s)
        i -= 1
    return "".join(out)


def _mul_small_add(m: list[int], k: int, add: int):
    carry = __bignative.mul_small_add(m, k, add)
    while carry > 0:
        m.append(carry & MASK)
        carry = carry >> LIMB


def _parse(text: str, base: int) -> object:
    s = text.strip()
    sign = 1
    if len(s) > 0 and (s[0] == "-" or s[0] == "+"):
        if s[0] == "-":
            sign = -1
        s = s[1:]
    if base == 0 or base == 16 or base == 8 or base == 2:
        if len(s) > 2 and s[0] == "0":
            p = s[1].lower()
            if p == "x" and base != 8 and base != 2:
                base = 16
                s = s[2:]
            elif p == "o" and base != 16 and base != 2:
                base = 8
                s = s[2:]
            elif p == "b" and base != 16 and base != 8:
                base = 2
                s = s[2:]
    if base == 0:
        base = 10
    if len(s) == 0 or s[0] == "_" or s[len(s) - 1] == "_" or "__" in s:
        raise ValueError("invalid literal for int() with base " + str(base) + ": " + repr(text))
    m: list[int] = []
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
        _mul_small_add(m, base, d)
    return _make(sign, m)


def _twos(sign: int, m: list[int], n: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < n:
        if i < len(m):
            out.append(m[i])
        else:
            out.append(0)
        i += 1
    if sign < 0:
        carry = 1
        i = 0
        while i < n:
            t = ((~out[i]) & MASK) + carry
            out[i] = t & MASK
            carry = t >> LIMB
            i += 1
    return out


def _bitop(op: int, a: object, b: object) -> object:
    if not _isint(a) or not _isint(b):
        raise TypeError("unsupported operand type(s) for bitwise operation")
    sa, ma = _parts(a)
    sb, mb = _parts(b)
    n = len(ma)
    if len(mb) > n:
        n = len(mb)
    n += 1
    xa = _twos(sa, ma, n)
    xb = _twos(sb, mb, n)
    out: list[int] = []
    i = 0
    while i < n:
        if op == 0:
            out.append(xa[i] & xb[i])
        elif op == 1:
            out.append(xa[i] | xb[i])
        else:
            out.append(xa[i] ^ xb[i])
        i += 1
    if (out[n - 1] >> (LIMB - 1)) & 1 == 1:
        carry = 1
        i = 0
        while i < n:
            t = ((~out[i]) & MASK) + carry
            out[i] = t & MASK
            carry = t >> LIMB
            i += 1
        return _make(-1, out)
    return _make(1, out)


def _rshift(a: object, n: object) -> object:
    if not _isint(a) or not isinstance(n, int):
        raise TypeError("unsupported operand type(s) for >>")
    if n < 0:
        raise ValueError("negative shift count")
    sa, ma = _parts(a)
    q = _shr_mag(ma, n)
    if sa < 0:
        # Floor: a negative value rounds away from zero when bits are lost.
        back = _shl_mag(q, n)
        if _cmp_mag(back, ma) != 0:
            q = _add_mag(q, [1])
    return _make(sa, q)


def _format(x: BigInt, spec: str) -> str:
    sign = x.sign
    digits = _mag_str(x.mag)
    fill = " "
    align = ">"
    width = 0
    group = ""
    kind = "d"
    zero = False
    plus = ""
    i = 0
    if len(spec) > 1 and (spec[1] == "<" or spec[1] == ">" or spec[1] == "^"):
        fill = spec[0]
        align = spec[1]
        i = 2
    elif len(spec) > 0 and (spec[0] == "<" or spec[0] == ">" or spec[0] == "^"):
        align = spec[0]
        i = 1
    if i < len(spec) and (spec[i] == "+" or spec[i] == " "):
        plus = spec[i]
        i += 1
    if i < len(spec) and spec[i] == "0":
        zero = True
        i += 1
    w = 0
    while i < len(spec) and spec[i] >= "0" and spec[i] <= "9":
        w = w * 10 + (ord(spec[i]) - 48)
        i += 1
    width = w
    if i < len(spec) and (spec[i] == "," or spec[i] == "_"):
        group = spec[i]
        i += 1
    if i < len(spec):
        kind = spec[i]
        i += 1
    if i != len(spec):
        raise ValueError("Invalid format specifier")
    if kind == "x" or kind == "X" or kind == "o" or kind == "b":
        base = 16
        if kind == "o":
            base = 8
        elif kind == "b":
            base = 2
        digits = _digits_in(x.mag, base)
        if kind == "X":
            digits = digits.upper()
    elif kind != "d" and kind != "n":
        raise ValueError("Unknown format code '" + kind + "' for object of type 'int'")
    if group != "" and kind == "d":
        out: list[str] = []
        k = len(digits)
        j = 0
        while j < k:
            if j > 0 and (k - j) % 3 == 0:
                out.append(group)
            out.append(digits[j])
            j += 1
        digits = "".join(out)
    prefix = ""
    if sign < 0:
        prefix = "-"
    elif plus != "":
        prefix = plus
    body = digits
    if zero:
        pad = width - len(prefix) - len(body)
        if pad > 0:
            body = "0" * pad + body
        return prefix + body
    text = prefix + body
    pad = width - len(text)
    if pad <= 0:
        return text
    if align == "<":
        return text + fill * pad
    if align == "^":
        left = pad // 2
        return fill * left + text + fill * (pad - left)
    return fill * pad + text


def _digits_in(m: list[int], base: int) -> str:
    # A limb is a whole number of digits of base 16, 8 and 2.
    if len(m) == 0:
        return "0"
    width = 15
    if base == 8:
        width = 20
    elif base == 2:
        width = 60
    out: list[str] = []
    i = len(m) - 1
    while i >= 0:
        if base == 16:
            s = hex(m[i])[2:]
        elif base == 8:
            s = oct(m[i])[2:]
        else:
            s = bin(m[i])[2:]
        if i < len(m) - 1:
            s = "0" * (width - len(s)) + s
        out.append(s)
        i -= 1
    return "".join(out)


# What the operators of a program that opted in call.


def add(a, b):
    if isinstance(a, int) and isinstance(b, int):
        r = a + b
        if ((a ^ r) & (b ^ r)) < 0:
            return _addsub(a, b, False)
        return r
    return a + b


def sub(a, b):
    if isinstance(a, int) and isinstance(b, int):
        r = a - b
        if ((a ^ b) & (a ^ r)) < 0:
            return _addsub(a, b, True)
        return r
    return a - b


def mul(a, b):
    if isinstance(a, int) and isinstance(b, int):
        r = a * b
        if a == 0:
            return 0
        if a == -1:
            if b == MIN:
                return _mulvals(a, b)
            return r
        if r // a != b:
            return _mulvals(a, b)
        return r
    return a * b


def power(a, b):
    if _isint(a) and isinstance(b, int):
        if b < 0:
            return float(a) ** b
        result = 1
        base = a
        while b > 0:
            if b & 1 == 1:
                result = mul(result, base)
            b = b >> 1
            if b > 0:
                base = mul(base, base)
        return result
    return a ** b


def powmod(a, b, m):
    if not _isint(a) or not _isint(b) or not _isint(m):
        raise TypeError("pow() 3rd argument not allowed unless all arguments are integers")
    if m == 0:
        raise ValueError("pow() 3rd argument cannot be 0")
    sb, mb = _parts(b)
    if sb < 0:
        raise ValueError("base is not invertible for the given modulus")
    result = 1 % m
    base = a % m
    bits = _bits_of(mb)
    i = bits - 1
    while i >= 0:
        result = mul(result, result) % m
        if (mb[i // LIMB] >> (i % LIMB)) & 1 == 1:
            result = mul(result, base) % m
        i -= 1
    return result


def lshift(a, n):
    if isinstance(a, int) and isinstance(n, int):
        if n < 0:
            raise ValueError("negative shift count")
        if a == 0:
            return 0
        if n < 62:
            r = a << n
            if (r >> n) == a:
                return r
        sa, ma = _of_int(a)
        return _make(sa, _shl_mag(ma, n))
    if isinstance(a, BigInt) and isinstance(n, int):
        if n < 0:
            raise ValueError("negative shift count")
        return _make(a.sign, _shl_mag(a.mag, n))
    return a << n


def abs_(x):
    if isinstance(x, BigInt):
        return BigInt(1, x.mag)
    if isinstance(x, int):
        if x == MIN:
            return _make(1, [0, 8])
        return abs(x)
    return abs(x)


def neg(a):
    if isinstance(a, int):
        if a == MIN:
            return _make(1, [0, 8])
        return -a
    return -a


def floordiv(a, b):
    if isinstance(a, int) and isinstance(b, int):
        if b == -1 and a == MIN:
            return _make(1, [0, 8])
        return a // b
    return a // b


def _radix(x: BigInt, base: int, prefix: str) -> str:
    text = prefix + _digits_in(x.mag, base)
    if x.sign < 0:
        return "-" + text
    return text


def hex_(x):
    if isinstance(x, BigInt):
        return _radix(x, 16, "0x")
    return hex(x)


def oct_(x):
    if isinstance(x, BigInt):
        return _radix(x, 8, "0o")
    return oct(x)


def bin_(x):
    if isinstance(x, BigInt):
        return _radix(x, 2, "0b")
    return bin(x)


def to_int(x):
    if isinstance(x, str):
        return _parse(x, 10)
    if isinstance(x, float):
        return _from_float(x)
    if isinstance(x, BigInt):
        return x
    return int(x)


def to_int_base(x, base):
    if isinstance(x, str):
        return _parse(x, base)
    return int(x, base)


def divmod_(a, b):
    if isinstance(a, int) and isinstance(b, int):
        if b == 0:
            raise ZeroDivisionError("division by zero")
        if not (a == MIN and b == -1):
            return a // b, a % b
        return _divmodvals(a, b)
    if _isint(a) and _isint(b):
        return _divmodvals(a, b)
    return divmod(a, b)


def sum_(xs, start=0):
    total = start
    for x in xs:
        total = add(total, x)
    return total


def prod(xs, start=1):
    total = start
    for x in xs:
        total = mul(total, x)
    return total


def is_int(x):
    return isinstance(x, int) or isinstance(x, BigInt)


def gcd(a, b):
    a = abs_(a)
    b = abs_(b)
    while b != 0:
        t = a % b
        a = b
        b = t
    return a


def lcm(a, b):
    if a == 0 or b == 0:
        return 0
    return abs_(mul(a // gcd(a, b), b))


def factorial(n):
    if n < 0:
        raise ValueError("factorial() not defined for negative values")
    result = 1
    i = 2
    while i <= n:
        result = mul(result, i)
        i += 1
    return result


def isqrt(n):
    if n < 0:
        raise ValueError("isqrt() argument must be nonnegative")
    if n < 2:
        return n
    x = lshift(1, (_bits_of(_parts(n)[1]) + 1) // 2)
    while True:
        y = add(x, n // x) // 2
        if y >= x:
            return x
        x = y


def comb(n, k):
    if k < 0 or n < 0:
        raise ValueError("n and k must be non-negative integers")
    if k > n:
        return 0
    if k > n - k:
        k = n - k
    result = 1
    i = 1
    while i <= k:
        result = mul(result, add(sub(n, k), i)) // i
        i += 1
    return result
