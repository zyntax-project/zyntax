# Record-shaped dicts: a few dicts with the same fixed string-literal
# keys, read field by field in a hot loop and updated in place.
# Returns 119680000.

def main() -> int:
    points = []
    for i in range(64):
        points.append({"x": i, "y": i * 2, "z": 64 - i, "w": 1})
    s = 0
    for r in range(20000):
        for p in points:
            s += p["x"] * p["w"] + p["y"] - p["z"]
            p["w"] = (p["w"] + 1) % 5
    return s

print(main())
