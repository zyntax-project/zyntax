import io
import sys


def report(out):
    print("to a file", 42, file=out)
    out.write("raw text\n")


def main():
    buf = io.StringIO()
    report(buf)
    print(repr(buf.getvalue()))
    seeded = io.StringIO("start:")
    seeded.write("more")
    print(seeded.getvalue(), seeded.read())
    captured = io.StringIO()
    original = sys.stdout
    sys.stdout = captured
    print("hidden", 1, 2.5)
    print("x", end="")
    sys.stdout.write("y\n")
    sys.stdout.flush()
    sys.stdout = original
    print(repr(captured.getvalue()))
    text = "  \r\nline one\nline two\r\nlast\r"
    print(text.splitlines(), [l.strip("\r\n ") for l in text.splitlines()])
    print("xxhixx".strip("x"), "xxhixx".lstrip("x"), "xxhixx".rstrip("x"), "abc".strip("z"))
    print("héllo wörld".strip("hd"), "..a..".strip("."))
    print(repr("".strip("x")), repr("xxx".strip("x")))


main()
