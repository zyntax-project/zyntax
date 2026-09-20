import pyflate_fast
import hashlib
pyflate_fast._main()
data = open("interpreter.tar.bz2", "rb").read()
field = pyflate_fast.RBitfield(open("interpreter.tar.bz2", "rb"))
print(len(data), field.readbits(16), field.readbits(8), field.tell())
print(hashlib.md5(b"hello world").hexdigest(), pyflate_fast.reverse_bits(0b1011, 4))
print("ok")
