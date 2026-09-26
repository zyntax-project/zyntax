# Methods passed as values beside direct calls of them: the direct calls
# are typed, the passed values stay the methods they name.
from array import array


class Cipher(object):
    def __init__(self, key):
        self.key = key

    def step(self, block, k):
        for i in range(len(block)):
            block[i] = (block[i] + k + self.key) & 255

    def encrypt_block(self, block):
        self.step(block, 1)

    def decrypt_block(self, block):
        self.step(block, -1 - 2 * self.key)

    def scale(self, x):
        return x * self.key


class Mode(object):
    def __init__(self, cipher):
        self.cipher = cipher

    def each(self, data, block_func):
        for offset in range(0, len(data), 4):
            block = data[offset:offset + 4]
            block_func(block)
            data[offset:offset + 4] = block
        return data

    def encrypt(self, data):
        return self.each(array('B', data), self.cipher.encrypt_block)

    def decrypt(self, data):
        return self.each(array('B', data), self.cipher.decrypt_block)

    def direct(self, data):
        data = array('B', data)
        block = data[0:4]
        self.cipher.encrypt_block(block)
        return list(block)


c = Cipher(3)
m = Mode(c)
enc = m.encrypt(b"abcdefgh")
print(list(enc))
print(list(m.decrypt(bytes(enc))))
print(m.direct(b"wxyz"))

# Passed, called with typed and dynamic arguments, stored and passed to
# a builtin.
print(c.scale(2), c.scale(1.5), c.scale("ab"))


def apply(f, x):
    return f(x)


print(apply(c.scale, 4), apply(c.scale, "x"))
kept = [c.scale, c.encrypt_block]
print(kept[0](5))
print(list(map(c.scale, [1, 2, 3])))

# An argument list mutated in place keeps its identity.
xs = [1, 2, 3]
c.encrypt_block(xs)
print(xs)
