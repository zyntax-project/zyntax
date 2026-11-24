print(int("41", 16), int("ff", 16), int("FF", 16), int("101", 2), int("0x1f", 16), int("z", 36), int("17", 8), int(" 7 ", 10))
print(int("0b101", 0), int("0o17", 0), int("0xff", 0), int("42", 0), int("-0x10", 16), int("+7", 8), int("1_000", 10))
for bad in [("2", 2), ("g", 16), ("", 10), ("1__0", 10), ("x", 37)]:
    try:
        int(bad[0], bad[1])
    except ValueError as e:
        print("ValueError", e)
