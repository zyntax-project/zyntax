# Counting by string key: `d[k] = d.get(k, 0) + 1` over a stream of
# words drawn from a fixed vocabulary, then the counts read back. A
# lookup and an update by string key per word; the strings are built
# once, so the dict is the whole of the cost.
# Returns 1965000.

def main() -> int:
    vocab = []
    for i in range(4000):
        vocab.append("w" + str(i * 7919 % 100003))
    counts = {}
    n = len(vocab)
    j = 0
    for i in range(1500000):
        j = (j * 1103515245 + 12345) % 2147483648
        w = vocab[j % n]
        counts[w] = counts.get(w, 0) + 1
    total = 0
    most = 0
    for w in vocab:
        c = counts.get(w, 0)
        total += c
        if c > most:
            most = c
    return total + most * 1000 + len(counts)

print(main())
