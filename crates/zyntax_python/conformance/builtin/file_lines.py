import os
import tempfile
import time

path = os.path.join(tempfile.gettempdir(), "zy_lines_" + str(int(time.time() * 1000000)) + ".txt")
with open(path, "w") as f:
    f.write("alpha\nbeta\n\ngamma")
    f.writelines(["\nx", "y\n"])
with open(path) as f:
    for line in f:
        print(repr(line))
with open(path) as f:
    print([l.strip() for l in f])
with open(path) as f:
    print(repr(f.readline()), repr(f.readline()))
    print(repr(f.readline()), repr(f.read()), repr(f.readline()))
with open(path) as f:
    print(f.readlines())
with open(path) as f:
    n = 0
    for line in f:
        n += len(line)
    print(n)
with open(path) as f:
    print(sum(1 for _ in f), os.path.exists(path))
os.remove(path)
print(os.path.exists(path))
print(os.path.join("a", "b", "c"), os.path.join("/x", "y"), os.path.join("a", "/b", "c"), os.path.join("a/", "b"))
print(os.path.splitext("dir.d/file.tar.gz"), os.path.splitext("noext"), os.path.splitext(".hidden"), os.path.splitext("a/b.c/d"))
print(os.path.split("/a/b/c.txt"), os.path.split("file"), os.path.split("/"), os.path.isabs("/x"), os.path.isabs("x"))
print(os.path.normpath("a//b/./c/../d"), os.path.normpath("/../x"), os.path.normpath("../a/.."), os.path.normpath(""))
print(os.path.basename("/x/y.txt"), os.path.dirname("/x/y.txt"))
