# A default is one of the values a parameter takes, whether or not any
# call leaves it out.

def add_option(name, action="store", kind=None, default=None, dest=None):
    if dest is None:
        dest = name
        while dest.startswith("-"):
            dest = dest[1:]
    print(name, action, kind, default, dest)


class Parser:
    def add(self, name, kind=None, default=None):
        print(name, kind, default)
        return self


add_option("-n", action="store", kind="int", default=100, dest="num_runs")
add_option("--flag")
add_option("--x", kind="str", default="time")
p = Parser()
p.add("a", kind="int").add("b", default=2.5).add("c")
