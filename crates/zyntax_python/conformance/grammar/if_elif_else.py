# test_grammar.GrammarTests.test_if
def classify(x: int) -> str:
    if x < 0:
        return "negative"
    elif x == 0:
        return "zero"
    elif x < 10:
        return "small"
    else:
        return "large"

print(classify(-5))
print(classify(0))
print(classify(3))
print(classify(100))
if 1:
    print("one is true")
if 0:
    print("zero is true")
else:
    print("zero is false")
