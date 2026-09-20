import os

def header(w, h, maxval):
    return b'%i %i\n%i\n' % (w, h, maxval)

def pixel(c):
    return b'%c%c%c' % (c, c, c)

def main():
    magic = b'P6\n'
    print(magic)
    print(repr(magic), len(magic), magic[0], magic[-1])
    print(header(3, 2, 255))
    px = pixel(200)
    print(px, len(px), px[1])
    print(magic + px, b'ab' * 3, 2 * b'xy')
    print(magic == b'P6\n', magic != b'P6\n', magic == px)
    print(list(b'abc'), bytes([104, 105]), bytes(3), bytes())
    print(b'hello world'[6:], b'hello'[::-1], b'hello'[1:4])
    print(b'hi'.decode(), 'hé'.encode(), 'hé'.encode('utf-8').decode('utf-8'))
    total = 0
    for v in b'\x01\x02\x03':
        total += v
    print(total)
    print(b'%d|%5d|%-5d|%05d|%x|%X|%o|%s|%%' % (42, 42, 42, 42, 255, 255, 8, b'in'))
    print(b'%.2f %e' % (3.14159, 12345.678))
    print(type(magic) == bytes, isinstance(magic, bytes), isinstance(magic, str), bool(b''), bool(b'x'))
    d = {b'k': 1}
    print(d[b'k'], b'k' in d)
    items = [b'b', b'a']
    print(items, len(items[0]))

    path = 'conformance_bytes_test.bin'
    f = open(path, 'wb')
    f.write(magic)
    f.write(header(3, 2, 255))
    for i in range(3):
        f.write(pixel(i * 100))
    f.close()
    f = open(path, 'rb')
    data = f.read()
    f.close()
    print(len(data), data[:3], data[-3:], data)
    with open(path, 'wb') as out:
        out.write(b'with ')
        out.write(b'statement\n')
    with open(path, 'rb') as inp:
        print(inp.read())
    with open(path, 'w') as t:
        t.write('text mode\n')
        t.write('second line\n')
    with open(path) as t:
        print(t.read())
    os.remove(path)

main()


def positions():
    path = 'conformance_bytes_pos.bin'
    with open(path, 'wb') as f:
        f.write(b'0123456789')
    with open(path, 'rb') as f:
        print(f.read(3), f.read(4), f.read(), f.read(2))
    os.remove(path)


positions()
