import json_bench
import json
print(json.dumps(json_bench.SIMPLE[0]))
print(json.dumps(json_bench.NESTED[0]))
print(len(json.dumps(json_bench.HUGE[0])), json.dumps(json_bench.EMPTY[0]))
print(json.dumps([1, 2.5, None, True, False, "a\"b\\c\n", {"k": [1, {"z": -0.0}]}, (1, 2), 1e300, "h\u00e9llo \u0105"]))
print(len(json_bench.main(1)))
