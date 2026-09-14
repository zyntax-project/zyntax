# Records as dicts: a table of rows built from literals, filtered on a
# field, aggregated by a key through `dict.get`, then scanned for the
# best total. The commonest shape of a data script, and the one that
# leans hardest on the dict representation and on boxing.
# Returns 5997822.

def main() -> int:
    rows = []
    for i in range(600000):
        rows.append({"id": i, "group": i % 17, "score": (i * 7919) % 1000, "flag": i % 3 == 0})
    totals = {}
    count = 0
    for row in rows:
        if row["flag"] and row["score"] > 100:
            g = row["group"]
            totals[g] = totals.get(g, 0) + row["score"]
            count += 1
    best = 0
    for g in totals:
        if totals[g] > best:
            best = totals[g]
    return best + count

print(main())
