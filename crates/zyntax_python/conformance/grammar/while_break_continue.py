# test_grammar.GrammarTests.test_while / test_break_continue_loop
i = 0
while i < 10:
    i += 1
    if i % 2 == 0:
        continue
    if i > 7:
        break
    print(i)
print("done", i)
