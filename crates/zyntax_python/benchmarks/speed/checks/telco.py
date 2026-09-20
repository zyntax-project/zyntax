import telco
import hashlib
from decimal import Decimal, Context, ROUND_HALF_EVEN, ROUND_DOWN, getcontext
print(telco.run() >= 0)
out = open("telco.out", "rb").read()
print(len(out), hashlib.md5(out).hexdigest())
getcontext().rounding = ROUND_DOWN
r = Decimal('0.00894')
p = Context(rounding=ROUND_HALF_EVEN).quantize(r * 1542, Decimal('0.01'))
print(r * 1542, p, (p * Decimal("0.0675")).quantize(Decimal('0.01')), p + Decimal('0.93'))
