# `%` formatting with a literal format: ints, floats, strings, repr,
# width, precision, flags, and the single-value form.

x = 5
name = "ab"
print("Bad %d" % 4)
print("v=%d s=%s r=%r" % (x, name, name))
print("%5d|%-5d|%05d|%+d|% d" % (42, 42, 42, 42, 42))
print("%.2f|%8.3f|%-8.1f|%e|%g" % (3.14159, 3.14159, 3.14159, 31415.9, 0.0001))
print("%x %X %o %c %c %%" % (255, 255, 8, 65, "z"))
print("%s and %s" % (1.5, [1, 2]))
print("%10s|%-10s|%.2s" % ("hi", "hi", "hello"))
print("Total time for %d iterations: %.2f secs" % (10, 1.23456))
print("Average time per iteration: %.2f ms" % (1.23456 * 1000 / 10))


def label(id):
    return "id %d" % id


print(label(3), "%d" % 2.7, "%s" % None, "%s" % True, "100%%" % ())
