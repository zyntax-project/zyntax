# slicing an array through a dynamic value keeps its typecode,
# and array(code, x) copies bytes it is handed dynamically as it copies
# typed bytes
from array import array


def middle(seq):
    # seq is whatever it is handed: the slice is of the same kind
    return seq[1:3]


def every_other(seq):
    return seq[::2]


def reversed_tail(seq):
    return seq[-2:][::-1]


def as_bytes_array(x):
    # x arrives dynamically: bytes copy straight in, anything else
    # goes through its items
    return array('B', x)


def as_short_array(x):
    return array('h', x)


def xor_blocks(data, key):
    data = array('B', data)
    out = array('B')
    for offset in range(0, len(data), 4):
        block = data[offset:offset + 4]
        for i in range(len(block)):
            block[i] ^= key[i]
        out.extend(block)
    return out.tobytes()


things = [array('B', [1, 2, 3, 4, 5]), array('d', [0.5, 1.5, 2.5, 3.5]), [1, 2, 3, 4], ["a", "b", "c", "d"], (1, 2, 3, 4), "abcd", b"abcd"]
for thing in things:
    m = middle(thing)
    print(m, every_other(thing), reversed_tail(thing))
a = array('B', [9, 8, 7, 6])
s = middle(a)
s[0] = 0
print(a, s, s == array('B', [0, 7]), list(s), len(s))
print(as_bytes_array(b"AB\x00\xff"), as_bytes_array([1, 2]), as_bytes_array(array('B', [3])))
print(as_short_array(b"\x01\x00\xff\xff"), as_short_array((5, -6)))
print(xor_blocks(b"hello world!", array('B', [1, 2, 3, 4])))
print(xor_blocks(bytes([255, 0, 255, 0]), [255, 255, 255, 255]))
try:
    as_short_array(b"abc")
except ValueError as e:
    print("ValueError", e)
try:
    as_bytes_array([256])
except OverflowError as e:
    print("OverflowError")
