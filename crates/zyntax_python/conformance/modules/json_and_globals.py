import json

EMPTY = ({}, 3)
SIMPLE = ({'key1': 0, 'key2': True, 'key3': 'value'}, 2)
cases = ['EMPTY', 'SIMPLE']
counter = 0


def bump():
    global counter
    counter += 1


def main():
    for case in cases:
        data, count = globals()[case]
        print(case, json.dumps(data), count)
    bump()
    print(globals()['counter'], globals()['cases'])
    print(json.dumps([1, -2, 2.5, 1e300, -0.0, None, True, False]))
    print(json.dumps("quote\" backslash\\ newline\n tab\t bell\b ff\f unicode ą del\x7f"))
    print(json.dumps({"a": [1, {"b": (2, 3)}], "c": {}}), json.dumps([]), json.dumps(()))
    print(json.dumps(float("inf")), json.dumps(float("-inf")), json.dumps(float("nan")))
    try:
        globals()["missing"]
    except KeyError:
        print("KeyError")


main()
