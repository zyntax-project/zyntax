# Files and in-memory files keep a position that seek and tell move.
import io
import os
import tempfile

buf = io.StringIO()
buf.write("hello world")
print(buf.tell())
buf.seek(0)
print(buf.read(5), buf.tell())
buf.seek(6)
print(buf.read())
buf.seek(0)
buf.truncate(5)
print(repr(buf.getvalue()), buf.tell())
buf.seek(0, 2)
buf.write("!")
print(buf.getvalue())
buf.seek(0)
buf.truncate()
print(repr(buf.getvalue()))
print(buf.readable(), buf.writable(), buf.seekable(), buf.closed)
buf.close()
print(buf.closed)

path = os.path.join(tempfile.gettempdir(), "zypy_positions.txt")
with open(path, "w") as f:
    print(f.name == path, f.writable(), f.readable(), f.closed)
    f.write("abcdef")
    print(f.tell())
print(f.closed)
with open(path) as f:
    print(f.readable(), f.writable())
    f.seek(2)
    print(f.read(2), f.tell())
    f.seek(0)
    print(f.readline(), f.tell())
    f.seek(0, 2)
    print(f.tell(), repr(f.read()))
os.remove(path)

try:
    buf.tell()
except ValueError as e:
    print("ValueError", e)
