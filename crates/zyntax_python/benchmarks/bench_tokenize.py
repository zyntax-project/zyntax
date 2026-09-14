# Text handling: strings built from pieces, joined, split back into
# tokens, counted into a dict keyed by string, and formatted out again.
# String allocation, dict lookups by string key and `str(int)` are the
# whole of it.
# Returns 2729.

def main() -> int:
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    parts = []
    for i in range(300000):
        parts.append(words[i % 8] + str(i % 100))
    text = " ".join(parts)
    counts = {}
    for tok in text.split(" "):
        counts[tok] = counts.get(tok, 0) + 1
    out = []
    for key in counts:
        if counts[key] > 350:
            out.append(key + "=" + str(counts[key]))
    joined = ",".join(out)
    return len(joined) + len(counts)

print(main())
