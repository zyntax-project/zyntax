for t in ["a'b", 'a"b', "a'\"b", "tab\there", "nl\n", "cr\r", "bs\\", "\x00\x01\x1f\x7f\x80\x9f\xa0\xad", "ünï", "日本語", "​ ", "😀", "\U0010ffff", ""]:
    print(repr(t), t)
print(["it's", 'say "hi"'], {"k'": 'v"'})
