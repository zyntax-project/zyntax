# Int keys inserted, read and deleted in one sliding window: every step
# adds a key, deletes the one that fell out of the window and probes a
# key inside it and one outside. Deletion is as common as insertion, so
# what a delete costs shows.
# Returns 539099476750.

def main() -> int:
    window = 1000
    d = {}
    s = 0
    for i in range(600000):
        d[i] = i * 3
        if i >= window:
            del d[i - window]
        s += d.get(i - window // 2, 0)
        if (i - window - 7) in d:
            s += 1
    return s + len(d)

print(main())
