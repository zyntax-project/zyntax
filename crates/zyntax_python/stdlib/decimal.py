"""Decimal arithmetic on an integer coefficient and a power of ten.

A Decimal is `coefficient * 10 ** exponent`, the coefficient an int of
at most 18 digits: the precision this module carries, where the
standard one carries 28 and any more on request. Addition, subtraction
and multiplication are exact within it; division and quantize round by
the rounding mode in force. A result that would need more digits raises
OverflowError rather than lose them.
"""

MAX_DIGITS = 18

ROUND_DOWN = 'ROUND_DOWN'
ROUND_HALF_EVEN = 'ROUND_HALF_EVEN'
ROUND_HALF_UP = 'ROUND_HALF_UP'
ROUND_UP = 'ROUND_UP'
ROUND_FLOOR = 'ROUND_FLOOR'
ROUND_CEILING = 'ROUND_CEILING'
ROUND_HALF_DOWN = 'ROUND_HALF_DOWN'
ROUND_05UP = 'ROUND_05UP'


def _pow10(n):
    p = 1
    for i in range(n):
        p *= 10
    return p


def _digits(n):
    if n < 0:
        n = -n
    count = 1
    while n >= 10:
        n //= 10
        count += 1
    return count


def _fits(digits):
    if digits > MAX_DIGITS:
        raise OverflowError("Decimal precision beyond 18 digits is not supported")


def _round_div(coefficient, divisor, rounding):
    """coefficient / divisor rounded to an integer by `rounding`."""
    negative = coefficient < 0
    if negative:
        coefficient = -coefficient
    q = coefficient // divisor
    r = coefficient - q * divisor
    if r != 0:
        twice = 2 * r
        if rounding == ROUND_DOWN:
            pass
        elif rounding == ROUND_UP:
            q += 1
        elif rounding == ROUND_HALF_EVEN:
            if twice > divisor or (twice == divisor and q % 2 == 1):
                q += 1
        elif rounding == ROUND_HALF_UP:
            if twice >= divisor:
                q += 1
        elif rounding == ROUND_HALF_DOWN:
            if twice > divisor:
                q += 1
        elif rounding == ROUND_FLOOR:
            if negative:
                q += 1
        elif rounding == ROUND_CEILING:
            if not negative:
                q += 1
        elif rounding == ROUND_05UP:
            if q % 10 == 0 or q % 10 == 5:
                q += 1
    if negative:
        q = -q
    return q


class Context(object):
    def __init__(self, prec=28, rounding=ROUND_HALF_EVEN):
        self.prec = prec
        self.rounding = rounding

    def quantize(self, value, exp):
        return value._quantize(exp, self.rounding)

    def create_decimal(self, value):
        return Decimal(value)


_context = Context(28, ROUND_HALF_EVEN)


def getcontext():
    return _context


def setcontext(context):
    global _context
    _context = context


class Decimal(object):
    def __init__(self, value=0):
        self.coefficient = 0
        self.exponent = 0
        if isinstance(value, str):
            self._parse(value)
        elif isinstance(value, Decimal):
            self.coefficient = value.coefficient
            self.exponent = value.exponent
        else:
            self.coefficient = int(value)
            self.exponent = 0

    def _parse(self, text):
        text = text.strip()
        negative = False
        if text.startswith('-'):
            negative = True
            text = text[1:]
        elif text.startswith('+'):
            text = text[1:]
        exponent = 0
        at = text.find('e')
        if at < 0:
            at = text.find('E')
        if at >= 0:
            exponent = int(text[at + 1:])
            text = text[:at]
        dot = text.find('.')
        if dot >= 0:
            fraction = text[dot + 1:]
            text = text[:dot] + fraction
            exponent -= len(fraction)
        if text == '':
            coefficient = 0
        else:
            coefficient = int(text)
        if negative:
            coefficient = -coefficient
        self.coefficient = coefficient
        self.exponent = exponent

    def _make(self, coefficient, exponent):
        d = Decimal(0)
        d.coefficient = coefficient
        d.exponent = exponent
        return d

    def _rounded(self, coefficient, exponent):
        """A result rounded to the context's precision."""
        prec = _context.prec
        if prec > MAX_DIGITS:
            prec = MAX_DIGITS
        digits = _digits(coefficient)
        if digits > prec:
            drop = digits - prec
            coefficient = _round_div(coefficient, _pow10(drop), _context.rounding)
            exponent += drop
        return self._make(coefficient, exponent)

    def _aligned(self, other):
        """Both coefficients at the smaller exponent."""
        if self.exponent == other.exponent:
            return self.coefficient, other.coefficient, self.exponent
        if self.exponent < other.exponent:
            _fits(_digits(other.coefficient) + other.exponent - self.exponent)
            scale = _pow10(other.exponent - self.exponent)
            return self.coefficient, other.coefficient * scale, self.exponent
        _fits(_digits(self.coefficient) + self.exponent - other.exponent)
        scale = _pow10(self.exponent - other.exponent)
        return self.coefficient * scale, other.coefficient, other.exponent

    def _coerce(self, other):
        if isinstance(other, Decimal):
            return other
        return Decimal(other)

    def __add__(self, other):
        other = self._coerce(other)
        a, b, exponent = self._aligned(other)
        return self._rounded(a + b, exponent)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._coerce(other)
        a, b, exponent = self._aligned(other)
        return self._rounded(a - b, exponent)

    def __rsub__(self, other):
        return self._coerce(other).__sub__(self)

    def __mul__(self, other):
        other = self._coerce(other)
        _fits(_digits(self.coefficient) + _digits(other.coefficient))
        return self._rounded(self.coefficient * other.coefficient, self.exponent + other.exponent)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self._make(-self.coefficient, self.exponent)

    def __truediv__(self, other):
        other = self._coerce(other)
        if other.coefficient == 0:
            raise ZeroDivisionError("division by zero")
        prec = _context.prec
        if prec > MAX_DIGITS:
            prec = MAX_DIGITS
        # Scale the dividend so the quotient carries the precision, as
        # far as the coefficient's digits allow.
        shift = prec + _digits(other.coefficient) - _digits(self.coefficient) + 1
        if shift < 0:
            shift = 0
        if _digits(self.coefficient) + shift > MAX_DIGITS:
            shift = MAX_DIGITS - _digits(self.coefficient)
        scaled = self.coefficient * _pow10(shift)
        q = _round_div(scaled, other.coefficient, _context.rounding)
        exponent = self.exponent - other.exponent - shift
        # An exact quotient drops the trailing zeros the scaling added.
        while shift > 0 and q % 10 == 0 and q != 0:
            q //= 10
            exponent += 1
            shift -= 1
        return self._rounded(q, exponent)

    def _quantize(self, exp, rounding):
        target = exp.exponent
        if self.exponent == target:
            return self._make(self.coefficient, target)
        if self.exponent > target:
            _fits(_digits(self.coefficient) + self.exponent - target)
            return self._make(self.coefficient * _pow10(self.exponent - target), target)
        _fits(target - self.exponent)
        divisor = _pow10(target - self.exponent)
        return self._make(_round_div(self.coefficient, divisor, rounding), target)

    def quantize(self, exp, rounding=None):
        if rounding is None:
            rounding = _context.rounding
        return self._quantize(exp, rounding)

    def _compare(self, other):
        other = self._coerce(other)
        a, b, exponent = self._aligned(other)
        if a < b:
            return -1
        if a > b:
            return 1
        return 0

    def __eq__(self, other):
        return self._compare(other) == 0

    def __ne__(self, other):
        return self._compare(other) != 0

    def __lt__(self, other):
        return self._compare(other) < 0

    def __le__(self, other):
        return self._compare(other) <= 0

    def __gt__(self, other):
        return self._compare(other) > 0

    def __ge__(self, other):
        return self._compare(other) >= 0

    def __hash__(self):
        a = self.coefficient
        e = self.exponent
        while a != 0 and a % 10 == 0:
            a //= 10
            e += 1
        return hash(a) * 31 + e

    def __bool__(self):
        return self.coefficient != 0

    def __int__(self):
        if self.exponent >= 0:
            return self.coefficient * _pow10(self.exponent)
        return _round_div(self.coefficient, _pow10(-self.exponent), ROUND_DOWN)

    def __float__(self):
        return float(self.coefficient) * (10.0 ** self.exponent)

    def __str__(self):
        coefficient = self.coefficient
        negative = coefficient < 0
        if negative:
            coefficient = -coefficient
        digits = str(coefficient)
        exponent = self.exponent
        adjusted = len(digits) - 1 + exponent
        if exponent <= 0 and adjusted >= -6:
            point = len(digits) + exponent
            if exponent == 0:
                text = digits
            elif point > 0:
                text = digits[:point] + '.' + digits[point:]
            else:
                text = '0.' + '0' * (-point) + digits
        else:
            text = digits[0]
            if len(digits) > 1:
                text = text + '.' + digits[1:]
            if adjusted >= 0:
                text = text + 'E+' + str(adjusted)
            else:
                text = text + 'E' + str(adjusted)
        if negative:
            text = '-' + text
        return text

    def __repr__(self):
        return "Decimal('" + str(self) + "')"
