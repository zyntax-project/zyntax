from decimal import *
from struct import unpack, calcsize


def main():
    getcontext().rounding = ROUND_DOWN
    rates = list(map(Decimal, ('0.0013', '0.00894')))
    twodig = Decimal('0.01')
    banker = Context(rounding=ROUND_HALF_EVEN)
    total = Decimal("0")
    for n in [39, 1542, 7, 1000000]:
        r = rates[n & 1]
        p = banker.quantize(r * n, twodig)
        b = (p * Decimal("0.0675")).quantize(twodig)
        total += p + b
        print(n, r * n, p, b, total)
    print(Decimal('1e3'), Decimal('0.0000001'), Decimal('123.450'), -Decimal('0.05'), Decimal(5))
    print(Decimal('10') / Decimal('4'), Decimal('1') / Decimal('8'), Decimal('2') * 3, 3 * Decimal('2'))
    print(Decimal('1.5') < Decimal('2'), Decimal('1.50') == Decimal('1.5'), int(Decimal('7.9')), float(Decimal('0.25')))
    print(repr(Decimal('0.10')), str(Decimal('-1.5e-7')), bool(Decimal('0.0')), Decimal('0.00'))
    print(Decimal('1.25').quantize(Decimal('0.1')), Decimal('1.35').quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
    data = b"\x00\x00\x00\x00\x00\x00\x00'"
    n, = unpack('>Q', data)
    print(n, unpack('<HB', b"\x01\x02\x03"), unpack('>d', b"\x40\x09\x21\xfb\x54\x44\x2d\x18"))
    print(unpack('<i', b"\xff\xff\xff\xff"), unpack('>h', b"\x80\x00"), unpack('?3sx', b"\x01abc\x00"), calcsize('>QHb'))


main()
