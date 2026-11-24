# format(value, spec) with the spec written out, and format(value).

print(format(3.14159, ".2f"), format(2.5, "8.3f") + "|", format(1234567.891, ",.1f"), format(0.25, ".0%"))
print(format(255, "x"), format(255, "#x"), format(255, "08b"), format(-255, "o"), format(42, "+d"))
print(format("ab", ">6") + "|", format("ab", "<6") + "|", format("ab", "^6") + "|", format("ab", "*^7"))
print(format(5), format(2.5), format("text"), format(True), format(None), format([1, 2]))
print(format(1e10, "g"), format(1e-5, "e"), format(12, "5d") + "|", format(-3, "05d"))
try:
    format(1, "q")
except ValueError as e:
    print("ValueError")
